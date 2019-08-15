#include <cmath>
#include <iostream>
#include <type_traits>
#include <omp.h>

using Real = double;
constexpr size_t _NUMSTEPS = 501;
constexpr size_t _N = 512;

template<typename T, size_t L, size_t W, size_t H>
class grid{
  using self_t = grid<T, L, W, H>;
  static constexpr size_t data_count = L * W * H;

public:
  T& at(int i, int j, int k){
    return data[((i + L) % L) * W * H + ((j + W) % W) * H + ((k + H) % H)];
  }

  const T& at(int i, int j, int k) const{
    return data[((i + L) % L) * W * H + ((j + W) % W) * H + ((k + H) % H)];
  }

private:
  T data[data_count] = {};
};

template<typename T, size_t L, size_t W, size_t H>
std::ostream & operator<<(std::ostream &os, const grid<T, L, W, H> &gr){
  for(u_int i = 0; i < L; ++i){
    os << "{ ";
    for(u_int j = 0; j < W; ++j){
      os << std::endl << "    { ";
      for(u_int k = 0; k < H; ++k){
        os << gr.at(i, j, k) << " ";
      }
      os << "} ";
    }
    os << "}" << std::endl;
  }
  return os;
}

template<typename T, size_t L, size_t W, size_t H, size_t depth>
struct level{
  grid<T, L, W, H> dat;

  void coarsen();
  void refine();

  static_assert(!(L % 2) && !(W % 2) && !(H % 2), "Parity issue!");
  level<T, L / 2, W / 2, H / 2, depth - 1> down;
};

template<typename T, size_t L, size_t W, size_t H>
struct level<T, L, W, H, 0> {
  grid<T, L, W, H> dat;
};

template<typename T, size_t L, size_t W, size_t H>
void StepRed(const grid<T, L, W, H>& b, grid<T, L, W, H>& x, int steps){
  if(steps <= 0){
    return;
  } else{
    #pragma omp parallel for
    for(u_int i = 0; i < L; ++i){
      for(u_int j = 0; j < W; ++j){
        for(u_int k= (i+j) % 2; k < H; k += 2){
            x.at(i,j,k) = (b.at(i,j,k)
              + x.at(i+1,j,k) + x.at(i-1,j,k)
              + x.at(i,j+1,k) + x.at(i,j-1,k)
              + x.at(i,j,k+1) + x.at(i,j,k-1)) / 6;
        }
      }
    }
    StepBlack(b, x, steps - 1);
  }
}

template<typename T, size_t L, size_t W, size_t H>
void StepBlack(const grid<T, L, W, H>& b, grid<T, L, W, H>& x, int steps){
  if(steps <= 0){
    return;
  } else{
    #pragma omp parallel for
    for(u_int i = 0; i < L; ++i){
      for(u_int j = 0; j < W; ++j){
        for(u_int k= (i+j+1) % 2; k < H; k += 2){
            x.at(i,j,k) = (b.at(i,j,k)
              + x.at(i+1,j,k) + x.at(i-1,j,k)
              + x.at(i,j+1,k) + x.at(i,j-1,k)
              + x.at(i,j,k+1) + x.at(i,j,k-1)) / 6;
        }
      }
    }
    StepRed(b, x, steps - 1);
  }
}

template<typename T, size_t L, size_t W, size_t H>
void GetStatus(const grid<T, L, W, H>& b, const grid<T, L, W, H>& x){
  T record = 0;
  T dist = 0;
  #pragma omp parallel for reduction(+: dist) reduction(max: record)
  for(u_int i = 0; i < L; ++i){
    for(u_int j = 0; j < W; ++j){
      for(u_int k = 0; k < H; ++k){
        T val = std::abs(6 * x.at(i,j,k)
          - x.at(i+1,j,k) - x.at(i-1,j,k)
          - x.at(i,j+1,k) - x.at(i,j-1,k)
          - x.at(i,j,k+1) - x.at(i,j,k-1) - b.at(i,j,k));
        dist += val;
        if(val > record){
          record = val;
        }
      }
    }
  }
  std::cout << "L1: " << dist << " Linfty: " << record << std::endl;
}

template<typename T, size_t L, size_t W, size_t H, size_t depth>
void level<T, L, W, H, depth>::coarsen(){
  #pragma omp parallel for
  for(u_int i = 0; i < L; i += 2){
    for(u_int j = 0; j < W; j += 2){
      for(u_int k = 0; k < H; k += 2){
        down.dat.at(i / 2, j / 2, k / 2) = 0;
        for(int dx = -1; dx < 2; ++dx){
          for(int dy = -1; dy < 2; ++dy){
            for(int dz = -1; dz < 2; ++dz){
              down.dat.at(i / 2, j / 2, k / 2)
                += dat.at(i + dx, j + dy, k + dz) * (dx ? 1 : 2) * (dy ? 1 : 2) * (dz ? 1 : 2);
            }
          }
        }
        down.dat.at(i / 2, j / 2, k / 2) /= 64;
      }
    }
  }
}

template<typename T, size_t L, size_t W, size_t H, size_t depth>
void level<T, L, W, H, depth>::refine(){
  #pragma omp parallel for
  for(u_int i = 0; i < L / 2; ++i){
    for(u_int j = 0; j < W / 2; ++j){
      for(u_int k = 0; k < H / 2; ++k){
        dat.at(2*i, 2*j, 2*k) = down.dat.at(i,j,k);
        dat.at(2*i+1, 2*j, 2*k) = (down.dat.at(i,j,k) + down.dat.at(i+1,j,k)) / 2;
        dat.at(2*i, 2*j+1, 2*k) = (down.dat.at(i,j,k) + down.dat.at(i,j+1,k)) / 2;
        dat.at(2*i, 2*j, 2*k+1) = (down.dat.at(i,j,k) + down.dat.at(i,j,k+1)) / 2;
        dat.at(2*i, 2*j+1, 2*k+1) = (down.dat.at(i,j,k) + down.dat.at(i,j+1,k) + down.dat.at(i,j,k+1) + down.dat.at(i,j+1,k+1)) / 4;
        dat.at(2*i+1, 2*j, 2*k+1) = (down.dat.at(i,j,k) + down.dat.at(i+1,j,k) + down.dat.at(i,j,k+1) + down.dat.at(i+1,j,k+1)) / 4;
        dat.at(2*i+1, 2*j+1, 2*k) = (down.dat.at(i,j,k) + down.dat.at(i,j+1,k) + down.dat.at(i+1,j,k) + down.dat.at(i+1,j+1,k)) / 4;
        dat.at(2*i+1, 2*j+1, 2*k+1) = (
          down.dat.at(i,j,k) + down.dat.at(i,j+1,k) + down.dat.at(i+1,j,k) + down.dat.at(i,j,k+1)
          + down.dat.at(i,j+1,k+1) + down.dat.at(i+1,j,k+1) + down.dat.at(i+1,j+1,k) + down.dat.at(i+1,j+1,k+1)
        ) / 8;
      }
    }
  }
}

template<typename T, size_t L, size_t W, size_t H>
void Solve(level<T, L, W, H, 0>& b, level<T, L, W, H, 0>& x, int steps){
  StepRed(b.dat, x.dat, steps);
  std::cout << "Level: 0 ";
  GetStatus(b.dat, x.dat);
}

template<typename T, size_t L, size_t W, size_t H, size_t depth>
void Solve(level<T, L, W, H, depth>& b, level<T, L, W, H, depth>& x, int steps){
  b.coarsen();
  x.coarsen();
  Solve(b.down, x.down, steps);
  x.refine();
  StepRed(b.dat, x.dat, steps);
  std::cout << "Level: " << depth << " ";
  GetStatus(b.dat, x.dat);
}

int main(){
  using level_t = level<Real, _N, _N, _N, 5>;

  level_t& b = *(new level_t);
  b.dat.at(0,0,0) = 1;
  b.dat.at(128,128,128) = -1;

  level_t& x = *(new level_t);

  Solve(b, x, _NUMSTEPS);
}
