#include <cmath>
#include <iostream>
#include <type_traits>
#include <omp.h>
#include <utility>

using Real = double;
constexpr size_t _NUMSTEPS = 40;
constexpr size_t _NUMSLASH = 10;
constexpr size_t _N = 512;
constexpr size_t _LEVELS = 6;
constexpr size_t _X = 0;
constexpr size_t _Y = 0;
constexpr size_t _Z = 1;
constexpr Real _THRESH = 1e-15;

template<typename T, size_t D1, size_t D2, size_t D3>
class grid{
  using self_t = grid<T, D1, D2, D3>;
  static constexpr size_t data_count = D1 * D2 * D3;

public:
  inline T& at(int i, int j, int k){
    return data[((i + D1) % D1) * D2 * D3 + ((j + D2) % D2) * D3 + ((k + D3) % D3)];
  }

  inline const T& at(int i, int j, int k) const{
    return data[((i + D1) % D1) * D2 * D3 + ((j + D2) % D2) * D3 + ((k + D3) % D3)];
  }

  void setZero(){
    #pragma omp parallel for
    for(u_int i = 0; i < D1; ++i){
      for(u_int j = 0; j < D2; ++j){
        for(u_int k = 0; k < D3; ++k){
          at(i,j,k) = 0;
        }
      }
    }
  }

  grid(){
    setZero();
  }

private:
  T data[data_count];
};

template<typename T, size_t D1, size_t D2, size_t D3>
std::ostream & operator<<(std::ostream &os, const grid<T, D1, D2, D3> &gr){
  for(u_int i = 0; i < D1; ++i){
    os << "{ ";
    for(u_int j = 0; j < D2; ++j){
      os << std::endl << "    { ";
      for(u_int k = 0; k < D3; ++k){
        os << gr.at(i, j, k) << " ";
      }
      os << "} ";
    }
    os << "}" << std::endl;
  }
  return os;
}

template<typename T, size_t D1, size_t D2, size_t D3, size_t depth>
struct level{
  using self_t = level<T, D1, D2, D3, depth>;

  grid<T, D1, D2, D3> dat;

  void coarsen();
  void refine();

  static_assert(!(D1 % 2) && !(D2 % 2) && !(D3 % 2), "Parity issue!");
  level<T, D1 / 2, D2 / 2, D3 / 2, depth - 1> down;
};

template<typename T, size_t D1, size_t D2, size_t D3>
struct level<T, D1, D2, D3, 0> {
  grid<T, D1, D2, D3> dat;
};

template<typename T, size_t D1, size_t D2, size_t D3>
void StepRed(const grid<T, D1, D2, D3>& b, grid<T, D1, D2, D3>& x, int steps){
  if(steps <= 0){
    return;
  } else{
    #pragma omp parallel for
    for(u_int i = 0; i < D1; ++i){
      for(u_int j = 0; j < D2; ++j){
        for(u_int k= (i+j) % 2; k < D3; k += 2){
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

template<typename T, size_t D1, size_t D2, size_t D3>
void StepBlack(const grid<T, D1, D2, D3>& b, grid<T, D1, D2, D3>& x, int steps){
  if(steps <= 0){
    return;
  } else{
    #pragma omp parallel for
    for(u_int i = 0; i < D1; ++i){
      for(u_int j = 0; j < D2; ++j){
        for(u_int k= (i+j+1) % 2; k < D3; k += 2){
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

template<typename T, size_t D1, size_t D2, size_t D3>
auto GetStatus(const grid<T, D1, D2, D3>& b, const grid<T, D1, D2, D3>& x) -> T{
  T record = 0;
  T dist = 0;
  #pragma omp parallel for reduction(+: dist) reduction(max: record)
  for(u_int i = 0; i < D1; ++i){
    for(u_int j = 0; j < D2; ++j){
      for(u_int k = 0; k < D3; ++k){
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
  return record;
}

template<typename T, size_t D1, size_t D2, size_t D3>
void GetRemainder(const grid<T, D1, D2, D3>& b, const grid<T, D1, D2, D3>& x, grid<T, D1, D2, D3>& rem){
  #pragma omp parallel for
  for(u_int i = 0; i < D1; ++i){
    for(u_int j = 0; j < D2; ++j){
      for(u_int k = 0; k < D3; ++k){
        rem.at(i,j,k) = b.at(i,j,k) - 6 * x.at(i,j,k)
          + x.at(i+1,j,k) + x.at(i-1,j,k)
          + x.at(i,j+1,k) + x.at(i,j-1,k)
          + x.at(i,j,k+1) + x.at(i,j,k-1);
      }
    }
  }
}

template<typename T, size_t D1, size_t D2, size_t D3>
void Accumulate(grid<T, D1, D2, D3>& x, const grid<T, D1, D2, D3>& y){
  #pragma omp parallel for
  for(u_int i = 0; i < D1; ++i){
    for(u_int j = 0; j < D2; ++j){
      for(u_int k = 0; k < D3; ++k){
        x.at(i,j,k) += y.at(i,j,k);
      }
    }
  }
}

template<typename T, size_t D1, size_t D2, size_t D3, size_t depth>
void level<T, D1, D2, D3, depth>::coarsen(){
  #pragma omp parallel for
  for(u_int i = 0; i < D1; i += 2){
    for(u_int j = 0; j < D2; j += 2){
      for(u_int k = 0; k < D3; k += 2){
        down.dat.at(i / 2, j / 2, k / 2) = 0;
        for(int dx = -1; dx < 2; ++dx){
          for(int dy = -1; dy < 2; ++dy){
            for(int dz = -1; dz < 2; ++dz){
              down.dat.at(i / 2, j / 2, k / 2)
                += dat.at(i + dx, j + dy, k + dz) * (dx ? 1 : 2) * (dy ? 1 : 2) * (dz ? 1 : 2);
            }
          }
        }
        down.dat.at(i / 2, j / 2, k / 2) /= 64 / 4;
      }
    }
  }
}

template<typename T, size_t D1, size_t D2, size_t D3, size_t depth>
void level<T, D1, D2, D3, depth>::refine(){
  #pragma omp parallel for
  for(u_int i = 0; i < D1 / 2; ++i){
    for(u_int j = 0; j < D2 / 2; ++j){
      for(u_int k = 0; k < D3 / 2; ++k){
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

template<typename T, size_t D1, size_t D2, size_t D3>
void Solve(level<T, D1, D2, D3, 0>& b, level<T, D1, D2, D3, 0>& x, int steps){
  //NB: I'm setting x to zero at the bottom.
  x.dat.setZero();
  StepRed(b.dat, x.dat, steps);
  std::cout << "Level: 0 ";
  GetStatus(b.dat, x.dat);
}

template<typename T, size_t D1, size_t D2, size_t D3, size_t depth>
void Solve(level<T, D1, D2, D3, depth>& b, level<T, D1, D2, D3, depth>& x, int steps){
  b.coarsen();
  Solve(b.down, x.down, steps);
  x.refine();
  StepRed(b.dat, x.dat, steps);
  std::cout << "Level: " << depth << " ";
  GetStatus(b.dat, x.dat);
}

template<typename T, size_t D1, size_t D2, size_t D3, size_t depth>
void Slash(const level<T, D1, D2, D3, depth>& b, level<T, D1, D2, D3, depth>& x, int steps, int times, T thresh = 0){
  using level_t = level<T, D1, D2, D3, depth>;

  level_t& x_scrap = *(new level_t);
  level_t& b_scrap1 = *(new level_t);
  level_t& b_scrap2 = *(new level_t);

  GetRemainder(b.dat, x.dat, b_scrap1.dat);

  int K = 0;
  T error;

  do{
    std::cout << "Slash: " << K << std::endl;
    Solve(b_scrap1, x_scrap, steps);
    Accumulate(x.dat, x_scrap.dat);
    GetRemainder(b_scrap1.dat, x_scrap.dat, b_scrap2.dat);
    std::swap(b_scrap1, b_scrap2);
    std::cout << "Overall: ";
    error = GetStatus(b.dat, x.dat);
    ++K;
  } while(K < times && error > thresh);

  delete &x_scrap;
  delete &b_scrap1;
  delete &b_scrap2;
}

int main(){
  using level_t = level<Real, _N, _N, _N, _LEVELS>;

  level_t& b = *(new level_t);
  b.dat.at(0,0,0) = 1;
  b.dat.at(_X, _Y, _Z) = -1;

  level_t& x = *(new level_t);

  Slash(b, x, _NUMSTEPS, _NUMSLASH/*, _THRESH*/);
  delete &x;
  delete &b;
}
