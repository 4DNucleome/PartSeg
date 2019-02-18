#include <cstdint>
#include <cstdlib>
#include <vector>
#include <limits>
#include <stdexcept>
#include <utility>
#include <array>
#include <ostream>
#include "my_queue.h"

typedef uint16_t coord_type;

namespace
{
template <typename T, size_t K>
size_t inline calculate_position(std::array<T, K> coordinate, std::array<size_t, K> dimension_size)
{
  size_t pos = 0;
  for (size_t i = 0; i < K; i++)
  {
    pos += coordinate[i] * dimension_size[i];
  }
  return pos;
}
template <typename T, size_t K>
bool inline outside_bounds(std::array<T, K> coordinate, std::array<T, K> lower_bound, std::array<T, K> upper_bound)
{
  for (size_t i = 0; i < K; i++)
  {
    if ((lower_bound[i] > coordinate[i]) || (upper_bound[i] <= coordinate[i]))
      return true;
  }
  return false;
}
template <typename T, size_t K>
std::ostream &operator<<(std::ostream &stream, const std::array<T, K> &array)
{
  stream << "array(";
  for (size_t i = 0; i < K - 1; i++)
    stream << array[i] << ", ";
  stream << array[K - 1] << ")";
  return stream;
}

template <size_t K>
std::ostream &operator<<(std::ostream &stream, const std::array<char, K> &array)
{
  stream << "array(";
  for (size_t i = 0; i < K - 1; i++)
    stream << (int)array[i] << ", ";
  stream << (int)array[K - 1] << ")";
  return stream;
}

template <typename T>
std::ostream &operator<<(std::ostream &stream, const std::vector<T> &array)
{
  stream << "vector(";
  for (size_t i = 0; i < array.size() - 1; i++)
    stream << array[i] << ", ";
  stream << array.back() << ")";
  return stream;
}
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
template <>
std::ostream &operator<<(std::ostream &stream, const std::vector<unsigned char> &array)
{
  stream << "vector(";
  for (size_t i = 0; i < array.size(); i++)
    stream << (int)array[i] << ", ";
  stream << (int)array.back() << ")";
  return stream;
}
std::ostream &operator<<(std::ostream &stream, const std::vector<signed char> &array)
{
  stream << "vector(";
  for (size_t i = 0; i < array.size(); i++)
    stream << (int)array[i] << ", ";
  stream << (int)array.back() << ")";
  return stream;
}
#pragma GCC diagnostic pop

template<typename T, size_t K>
std::array<T, K> operator-(const std::array<T, K> & v1, const  std::array<T, K> & v2){
  std::array<T, K> res;
  for (size_t i = 0; i < K; i++)
    res[i] = v1[i] - v2[i];
  return res;
}

template<typename T, size_t K>
std::array<T, K> operator+(const std::array<T, K> & v1, const std::array<T, K> & v2){
  std::array<T, K> res;
  for (size_t i = 0; i < K; i++)
    res[i] = v1[i] + v2[i];
  return res;
}

template <typename T, size_t K>
class ArrayLimits
{
  std::array<T, K> lower_bound;
  std::array<T, K> upper_bound;

public:
  ArrayLimits()
  {
    this->lower_bound.fill(0);
    this->upper_bound.fill(1);
  };
  ArrayLimits(std::array<T, K> lower_bound_, std::array<T, K> upper_bound_) : lower_bound(lower_bound_), upper_bound(upper_bound_){};
  ArrayLimits(std::array<T, K> upper_bound_) : upper_bound(upper_bound_)
  {
    this->lower_bound.fill(0);
  };
  void set_bounds(std::array<T, K> lower_bound, std::array<T, K> upper_bound)
  {
    this->lower_bound = lower_bound;
    this->upper_bound = upper_bound;
  };

  size_t size(){
    size_t res = 1;
    for (auto el: this->upper_bound - this->lower_bound)
      res *= el;
    return res;
  }

  class iterator
  {
  private:
    std::array<T, K> lower_bound;
    std::array<T, K> upper_bound;
    std::array<T, K> state;

  public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = std::array<T, K>;
    using difference_type = int;
    using pointer = std::array<T, K> *;
    using reference = std::array<T, K> &;

    iterator(std::array<T, K> lower_bound, std::array<T, K> upper_bound)
    {
      this->lower_bound = lower_bound;
      this->upper_bound = upper_bound;
      this->upper_bound[0] += 1;
      this->state = lower_bound;
    }
    iterator(std::array<T, K> lower_bound, std::array<T, K> upper_bound, std::array<T, K> state)
    {
      this->lower_bound = lower_bound;
      this->upper_bound = upper_bound;
      this->upper_bound[0] += 1;
      this->state = state;
    }
    std::array<T, K> operator++()
    {
      for (size_t i = K - 1; i >= 0; i--)
      {
        this->state[i]++;
        if (this->state[i] >= this->upper_bound[i])
          this->state[i] = this->lower_bound[i];
        else
          break;
      }
      return this->state;
    }
    std::array<T, K> operator++(int)
    {
      const auto res = this->state;
      ++this;
      return res;
    }
    std::array<T, K> operator--()
    {
      for (size_t i = K - 1; i >= 0; i--)
      {
        if (this->state[i] > this->lower_bound[i])
        {
          this->state[i]--;
          break;
        }
        else
        {
          this->state[i] = this->upper_bound[i] - 1;
        }
      }
      return this->state;
    }
    std::array<T, K> operator--(int)
    {
      const auto res = this->state;
      --this;
      return res;
    }
    std::array<T, K> operator*() { return this->state; };
    pointer operator->() { return &(this->state); };
    bool operator==(const iterator &other) const { return this->state == other.state; }
    bool operator!=(const iterator &other) const { return this->state != other.state; }
  };

  iterator begin() { return iterator(this->lower_bound, this->upper_bound); };
  iterator end()
  {
    std::array<T, K> state;
    state.fill(0);
    state[0] = this->upper_bound[0];
    return iterator(this->lower_bound, this->upper_bound, state);
  };
};

} // namespace

namespace MSO
{
template <typename T, typename M = double, size_t N=3>
/* K is number of dimensions */
class MSO
{
public:
  static const size_t ndim = N;
  typedef std::array<coord_type, N> Point;
  typedef M mu_type;
private:
  
  std::vector<int8_t> neighbourhood;
  std::vector<mu_type> distances;
  std::vector<mu_type> mu_array;
  std::array<coord_type, ndim> size;
  Point lower_bound;
  Point upper_bound;
  std::vector<mu_type> fdt_array;
  T components_num;
  bool use_background = false;
  T *components;
  const T background_component = 1;

public:
  MSO()
  {
    this->components = nullptr;
    this->size = {0};
  };

  void set_use_background(bool val)
  {
    this->use_background = val;
  }

  void set_components_num(T components_num){
    this->components_num = components_num;
  }

  void erase_data()
  {
    /* clean pointers, do not free the memory */
    this->components = nullptr;
    this->size.fill(0);
  }

  inline size_t get_length() const
  {
    size_t res = 1;
    for (size_t i = 0; i < ndim; i++)
      res *= this->size[i];
    return res;
  }

  inline std::array<size_t, ndim> dimension_size() const
  {
    std::array<size_t, ndim> res;
    res[ndim - 1] = 1;
    for (size_t i = ndim - 1; i > 0; i--)
    {
      res[i - 1] = res[i] * this->size[i];
    }
    return res;
  }

  template <typename W>
  void set_data(T *components, W size)
  {
    this->components = components;
    for (size_t i = 0; i < ndim; i++)
    {
      this->size[i] = size[i];
      this->upper_bound[i] = size[i];
      this->lower_bound[i] = 0;
    }
    if (this->get_length() != this->mu_array.size())
      this->mu_array.clear();
  }

  template <typename W>
  void set_bounding_box(W lower_bound, W upper_bound)
  {
    for (size_t i = 0; i < ndim; i++)
    {
      this->lower_bound[i] = lower_bound[i];
      this->upper_bound[i] = upper_bound[i];
    }
  }

  void set_mu_copy(const std::vector<mu_type> &mu)
  {
    if (mu.size() != this->get_length())
      throw std::length_error("Size of mu array need to be equal to size of components (z_size * y_size * x_size)");
    this->mu_array = mu;
  }
  void set_mu_copy(mu_type *mu, size_t length)
  {
    if (length != this->get_length())
      throw std::length_error("Size of mu array need to be equal to size of components (z_size * y_size * x_size)");
    this->mu_array = std::vector<mu_type>(mu, mu + length);
  }

  void set_mu_swap(std::vector<mu_type> &mu)
  {
    if (mu.size() != this->get_length())
      throw std::length_error("Size of mu array need to be equal to size of components (z_size * y_size * x_size)");
    this->mu_array.swap(mu);
  }

  void set_neighbourhood(std::vector<int8_t> neighbourhood, std::vector<mu_type> distances)
  {
    if (neighbourhood.size() != ndim * distances.size())
    {
      throw std::length_error("Size of neighbouthood need to be 3* Size of distances");
    }
    this->neighbourhood = neighbourhood;
    this->distances = distances;
  }

  void set_neighbourhood(int8_t *neighbourhood, mu_type *distances, size_t neigh_size)
  {
    this->neighbourhood = std::vector<int8_t>(neighbourhood, neighbourhood + 3 * neigh_size);
    this->distances = std::vector<double>(distances, distances + neigh_size);
  }

  void compute_FDT(std::vector<mu_type> &array) const
  {
    if (this->get_length() == 0)
      throw std::runtime_error("call FDT calculation befor set coordinates data");
    if (this->mu_array.size() == 0)
      throw std::runtime_error("call FDT calculation befor set mu array");

    const std::array<size_t, ndim> dimension_size = this->dimension_size();
    Point coord, coord2;
    size_t position, neigh_position, array_position, array_neigh_position;
    my_queue<Point> queue;
    double val, mu_value, fdt_value;
    std::vector<bool> visited_array(this->get_length(), false);
    //std::cout << "Neighbourhood: " << this->neighbourhood << std::endl << "Distances: " << this->distances << std::endl;
    auto bounds = ArrayLimits<coord_type, ndim>(this->lower_bound, this->upper_bound);
    for (auto coord : bounds)
    {
      position = calculate_position(coord, dimension_size);
      array_position = calculate_position(coord - this->lower_bound, dimension_size);
      array[array_position] = std::numeric_limits<mu_type>::infinity();
      if (this->components[position] == this->background_component)
      {
        array[position] = 0;
        for (size_t i = 0; i < 3 * this->distances.size(); i += 3)
        {
          for (size_t j = 0; j < ndim; j++)
            coord2[j] = coord[j] + this->neighbourhood[i + j];
          if (outside_bounds(coord2, lower_bound, upper_bound))
          {
            continue;
          }
          if (components[calculate_position(coord2, dimension_size)] == 0)
          {
            queue.push(coord);
            break;
          }
        }
      }
    }
    //std::cout << std::endl << "Queue size " << queue.get_size() << std::endl;
    size_t count = 0;
    while (!queue.empty())
    {
      count += 1;
      coord = queue.front();
      queue.pop();
      position = calculate_position(coord, dimension_size);
      array_position = calculate_position(coord - this->lower_bound, dimension_size);
      mu_value = this->mu_array[position];
      fdt_value = array[array_position];
      for (size_t i = 0; i < this->distances.size(); i++)
      {
        for (size_t j = 0; j < ndim; j++)
          coord2[j] = coord[j] + this->neighbourhood[3 * i + j];
        if (outside_bounds(coord2, lower_bound, upper_bound))
          continue;
        neigh_position = calculate_position(coord2, dimension_size);
        array_neigh_position = calculate_position(coord2 - this->lower_bound, dimension_size);
        if (this->components[neigh_position] != 0)
          continue;
        val = (this->mu_array[neigh_position] + mu_value) * distances[i] / 2;
        if (array[array_neigh_position] > val + fdt_value)
        {
          array[array_neigh_position] = val + fdt_value;
          if (!visited_array[neigh_position])
          {
            visited_array[neigh_position] = true;
            queue.push(coord2);
          }
        }
      }
      visited_array[position] = false;
    }
    //std::cout << "Count " << count << std::endl;
  };

  size_t optimum_erosion_calculate(std::vector<mu_type> &fdt_array, std::vector<T> &components_arr, std::vector<bool> sprawl_area)
  {
    Point coord, coord2;
    size_t position, neigh_position;
    mu_type val, val2;
    std::vector<mu_type> distances(fdt_array.size(), 0);
    std::vector<my_queue<Point>> queues(this->components_num + 1);
    std::vector<bool> visited_array(this->get_length(), false);
    auto bounds = ArrayLimits<coord_type, ndim>(this->upper_bound - this->lower_bound);
    const std::array<size_t, ndim> dimension_size = this->dimension_size();
    for (auto coord : bounds)
    {
      position = calculate_position(coord, dimension_size);
      if (components_arr[position] != 0)
      {
        distances[position] = fdt_array[position];
        for (size_t i = 0; i < 3 * this->distances.size(); i += 3)
        {
          for (size_t j = 0; j < ndim; j++)
            coord2[j] = coord[j] + this->neighbourhood[i + j];
          if (outside_bounds(coord2, lower_bound, upper_bound))
            continue;
          if (sprawl_area[calculate_position(coord2, dimension_size)] == true)
          {
            queues[components_arr[position]].push(coord);
            break;
          }
        }
      }
    }
    for (auto &queue : queues)
    {
      while (!queue.empty())
      {
        coord = queue.front();
        queue.pop();
        position = calculate_position(coord, dimension_size);
        val = distances[position];
        for (size_t i = 0; i < this->distances.size(); i++)
        {
          for (size_t j = 0; j < ndim; j++)
            coord2[j] = coord[j] + this->neighbourhood[3 * i + j];
          if (outside_bounds(coord2, lower_bound, upper_bound))
            continue;
          neigh_position = calculate_position(coord2, dimension_size);
          if (sprawl_area[neigh_position] == false)
            continue;
          val2 = std::min(val, fdt_array[neigh_position]);
          if (val2 < distances[neigh_position])
            continue;
          if ((val2 == distances[neigh_position]) && ((components_arr[neigh_position] == components_arr[position]) || (components_arr[neigh_position] == std::numeric_limits<T>::max())))
            continue;
          if (val2 > distances[neigh_position])
          {
            distances[neigh_position] = val2;
            components_arr[neigh_position] = components_arr[position];
          } else {
            components_arr[neigh_position] = std::numeric_limits<T>::max();
          }
          if (!visited_array[neigh_position])
          {
            visited_array[neigh_position] = true;
            queue.push(coord2);
          }
        }
        visited_array[neigh_position] = false;
      }
    }
    size_t count = 0;
    for (auto & el: components_arr){
      if (el == std::numeric_limits<T>::max()){
        el = 0;
      }
    }
    for (size_t i=0; i < bounds.size(); i++){
      if (sprawl_area[i] && components_arr[i] > 0){
        sprawl_area[i] = false;
        count++;
      }
    }
    return count;
  };
};

template <typename R>
void inline shrink(R &val)
{
  if (val > 1)
    val = 1;
  else if (val < 0)
    val = 0;
}

template <typename R, typename T>
class MuCalc{
  public:

  static std::vector<R> calculate_mu_array(T *array, size_t length, T lower_bound,
                                          T upper_bound)
  {
    std::vector<R> result(length, 0);
    R mu;
    for (size_t i = 0; i < length; i++)
    {
      mu = (R)(array[i] - lower_bound) / (upper_bound - lower_bound);
      shrink(mu);
      result[i] = mu;
    }
    return result;
  }

  static std::vector<R>  calculate_reflection_mu_array(T *array, size_t length, T lower_bound,
                                                    T upper_bound)
  {
    std::vector<R> result(length, 0);
    R mu;
    for (size_t i = 0; i < length; i++)
    {
      mu = (R)(array[i] - lower_bound) / (upper_bound - lower_bound);
      shrink(mu);
      if (mu < 0.5)
        mu = 1 - mu;
      result[i] = mu;
    }
    return result;
  }

  static std::vector<R>  calculate_two_object_mu(T *array, size_t length, T lower_bound,
                                              T upper_bound, T lower_mid_bound,
                                              T upper_mid_bound)
  {
    std::vector<R> result(length, 0);
    R mu;
    T pixel_val;
    for (size_t i = 0; i < length; i++)
    {
      pixel_val = array[i];
      mu = (R)(pixel_val - lower_bound) / (upper_bound - lower_bound);
      if (((lower_bound - lower_mid_bound) > 0) &&
          (pixel_val >= lower_mid_bound) && (pixel_val <= lower_bound))
        mu = (R)(pixel_val - lower_mid_bound) / (lower_bound - lower_mid_bound);
      else if (((upper_bound - lower_bound) > 0) && (lower_bound < pixel_val) &&
              (pixel_val <= upper_bound))
        mu = (R)(upper_bound - pixel_val) / (upper_bound - lower_bound);
      shrink(mu);
      result[i] = mu;
    }
    return result;
  }

  static std::vector<R>  calculate_mu_array_masked(T *array, size_t length, T lower_bound,
                                                T upper_bound, uint8_t *mask)
  {
    std::vector<R> result(length, 0);
    R mu;
    for (size_t i = 0; i < length; i++)
    {
      if (mask[i] == 0)
        continue;
      mu = (R)(array[i] - lower_bound) / (upper_bound - lower_bound);
      shrink(mu);
      result[i] = mu;
    }
    return result;
  }

  static std::vector<R>  calculate_reflection_mu_array_masked(T *array, size_t length,
                                                            T lower_bound, T upper_bound,
                                                            uint8_t *mask)
  {
    std::vector<R> result(length, 0);
    R mu;
    for (size_t i = 0; i < length; i++)
    {
      if (mask[i] == 0)
        continue;
      mu = (R)(array[i] - lower_bound) / (upper_bound - lower_bound);
      shrink(mu);
      if (mu < 0.5)
        mu = 1 - mu;
      result[i] = mu;
    }
    return result;
  }
  static std::vector<R>  calculate_two_object_mu_masked(T *array, size_t length, T lower_bound,
                                                      T upper_bound, T lower_mid_bound,
                                                      T upper_mid_bound, uint8_t *mask)
  {
    std::vector<R> result(length, 0);
    R mu;
    T pixel_val;
    for (size_t i = 0; i < length; i++)
    {
      if (mask[i] == 0)
        continue;
      pixel_val = array[i];
      mu = (R)(pixel_val - lower_bound) / (upper_bound - lower_bound);
      if (((lower_bound - lower_mid_bound) > 0) &&
          (pixel_val >= lower_mid_bound) && (pixel_val <= lower_bound))
        mu = (R)(pixel_val - lower_mid_bound) / (lower_bound - lower_mid_bound);
      else if (((upper_bound - lower_bound) > 0) && (lower_bound < pixel_val) &&
              (pixel_val <= upper_bound))
        mu = (R)(upper_bound - pixel_val) / (upper_bound - lower_bound);
      shrink(mu);
      result[i] = mu;
    }
    return result;
}
};
} // namespace MSO
