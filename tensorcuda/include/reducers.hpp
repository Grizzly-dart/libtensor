#ifndef REDUCERS_HPP
#define REDUCERS_HPP

template <typename T>
class Mean {
 public:
  T mean = 0;
  uint32_t n = 0;

  __device__ void consume(T sample);

  __device__ void merge(const Mean<T>& other);

  __device__ Mean<T> shfl_down(int offset);
};

template class Mean<double>;
template class Mean<float>;

template <typename T>
class Variance {
 public:
  T mean = 0;
  uint32_t n = 0;
  T m2 = 0;

  __device__ void consume(T sample);

  __device__ void merge(const Variance<T>& other);

  __device__ Variance<T> shfl_down(int offset);
};

template class Variance<double>;
template class Variance<float>;

#endif // REDUCERS_HPP
