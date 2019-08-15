#include <cmath>
#include <iostream>
#include <type_traits>
//#include <omp.h>

using Real = double;
constexpr size_t _NUMSTEPS = 101;
constexpr int _PRINT_EVERY = 10;
constexpr size_t _N = 32;

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
  T data[data_count];
};

template<typename T, size_t L, size_t W, size_t H, size_t depth>
struct level{
  grid<T, L, W, H> b;
  grid<T, L, W, H> x;

  static_assert(!(L % 2) && !(W % 2) && !(H % 2), "Parity issue!");
  level<T, L / 2, W / 2, H / 2, depth - 1> down;
};

template<typename T, size_t L, size_t W, size_t H>
struct level<T, L, W, H, 0> {
  grid<T, L, W, H> b;
  grid<T, L, W, H> x;
};

template<typename T, size_t L, size_t W, size_t H>
void StepRed(const grid<T, L, W, H>& b, grid<T, L, W, H>& x, int steps){
  if(steps >= _NUMSTEPS){
    return;
  } else{
    if(!(steps % _PRINT_EVERY)){
      std::cout << "Step: " << steps << " ";
      GetStatus(b, x);
    }

    // #pragma omp parallel for
    for(int i = 0; i < L; ++i){
      for(int j = 0; j < W; ++j){
        for(int k= (i+j) % 2; k < H; k += 2){
            x.at(i,j,k) = (b.at(i,j,k)
              + x.at(i+1,j,k) + x.at(i-1,j,k)
              + x.at(i,j+1,k) + x.at(i,j-1,k)
              + x.at(i,j,k+1) + x.at(i,j,k-1)) / 6;
        }
      }
    }
    StepBlack(b, x, steps + 1);
  }
}

template<typename T, size_t L, size_t W, size_t H>
void StepBlack(const grid<T, L, W, H>& b, grid<T, L, W, H>& x, int steps){
  if(steps >= _NUMSTEPS){
    return;
  } else{
    if(!(steps % _PRINT_EVERY)){
      std::cout << "Step: " << steps << " ";
      GetStatus(b, x);
    }

    // #pragma omp parallel for
    for(int i = 0; i < L; ++i){
      for(int j = 0; j < W; ++j){
        for(int k= (i+j+1) % 2; k < H; k += 2){
            x.at(i,j,k) = (b.at(i,j,k)
              + x.at(i+1,j,k) + x.at(i-1,j,k)
              + x.at(i,j+1,k) + x.at(i,j-1,k)
              + x.at(i,j,k+1) + x.at(i,j,k-1)) / 6;
        }
      }
    }
    StepRed(b, x, steps + 1);
  }
}

template<typename T, size_t L, size_t W, size_t H>
void GetStatus(const grid<T, L, W, H>& b, const grid<T, L, W, H>& x){
  T record = 0;
  T dist = 0;
  // #pragma omp parallel for reduction(+: dist) reduction(max: record)
  for(int i = 0; i < L; ++i){
    for(int j = 0; j < W; ++j){
      for(int k = 0; k < H; ++k){
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

int main(){
  level<Real, _N, _N, _N, 0> data = {};
  data.b.at(0,0,0) = 1;
  data.b.at(0,0,1) = -1;

  StepRed(data.b, data.x, 0);
}
