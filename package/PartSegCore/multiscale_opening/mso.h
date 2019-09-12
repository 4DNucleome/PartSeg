#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <ostream>
#include <stdexcept>
#include <utility>
#include <vector>
#include "my_queue.h"

typedef uint16_t coord_type;

namespace {
template <typename T, size_t K>
size_t inline calculate_position(std::array<T, K> coordinate,
                                 std::array<size_t, K> dimension_size) {
  size_t pos = 0;
  for (size_t i = 0; i < K; i++) {
    pos += coordinate[i] * dimension_size[i];
  }
  return pos;
}
template <typename T, size_t K>
bool inline outside_bounds(std::array<T, K> coordinate,
                           std::array<T, K> lower_bound,
                           std::array<T, K> upper_bound) {
  for (size_t i = 0; i < K; i++) {
    if ((lower_bound[i] > coordinate[i]) || (upper_bound[i] <= coordinate[i]))
      return true;
  }
  return false;
}
template <typename T, size_t K>
std::ostream &operator<<(std::ostream &stream, const std::array<T, K> &array) {
  stream << "array(";
  for (size_t i = 0; i < K - 1; i++) stream << array[i] << ", ";
  stream << array[K - 1] << ")";
  return stream;
}

template <size_t K>
std::ostream &operator<<(std::ostream &stream,
                         const std::array<char, K> &array) {
  stream << "array(";
  for (size_t i = 0; i < K - 1; i++) stream << (int)array[i] << ", ";
  stream << (int)array[K - 1] << ")";
  return stream;
}

template <typename T>
std::ostream &operator<<(std::ostream &stream, const std::vector<T> &array) {
  stream << "vector(";
  for (size_t i = 0; i < array.size() - 1; i++) stream << array[i] << ", ";
  stream << array.back() << ")";
  return stream;
}
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
template <>
std::ostream &operator<<(std::ostream &stream,
                         const std::vector<unsigned char> &array) {
  stream << "vector(";
  for (size_t i = 0; i < array.size(); i++) stream << (int)array[i] << ", ";
  stream << (int)array.back() << ")";
  return stream;
}
std::ostream &operator<<(std::ostream &stream,
                         const std::vector<signed char> &array) {
  stream << "vector(";
  for (size_t i = 0; i < array.size(); i++) stream << (int)array[i] << ", ";
  stream << (int)array.back() << ")";
  return stream;
}
#pragma GCC diagnostic pop

template <typename T, size_t K>
std::array<T, K> operator-(const std::array<T, K> &v1,
                           const std::array<T, K> &v2) {
  std::array<T, K> res;
  for (size_t i = 0; i < K; i++) res[i] = v1[i] - v2[i];
  return res;
}

template <typename T, size_t K>
std::array<T, K> operator+(const std::array<T, K> &v1,
                           const std::array<T, K> &v2) {
  std::array<T, K> res;
  for (size_t i = 0; i < K; i++) res[i] = v1[i] + v2[i];
  return res;
}

template <typename T, size_t K>
std::array<size_t, K> calculate_dimension_size(const std::array<T, K> &size) {
  std::array<size_t, K> res;
  res[K - 1] = 1;
  for (size_t i = K - 1; i > 0; i--) {
    res[i - 1] = res[i] * size[i];
  }
  return res;
};

template <typename T, size_t K>
size_t calculate_area_size(const std::array<T, K> &size) {
  size_t res = 1;
  for (size_t i = 0; i < K; i++) res *= size[i];
  return res;
};

template <typename T, size_t K>
class ArrayLimits {
  std::array<T, K> lower_bound;
  std::array<T, K> upper_bound;

 public:
  ArrayLimits() {
    this->lower_bound.fill(0);
    this->upper_bound.fill(1);
  };
  ArrayLimits(std::array<T, K> lower_bound_, std::array<T, K> upper_bound_)
      : lower_bound(lower_bound_), upper_bound(upper_bound_){};
  ArrayLimits(std::array<T, K> upper_bound_) : upper_bound(upper_bound_) {
    this->lower_bound.fill(0);
  };
  void set_bounds(std::array<T, K> lower_bound, std::array<T, K> upper_bound) {
    this->lower_bound = lower_bound;
    this->upper_bound = upper_bound;
  };

  size_t size() {
    size_t res = 1;
    for (auto el : this->upper_bound - this->lower_bound) res *= el;
    return res;
  }

  class iterator {
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

    iterator(std::array<T, K> lower_bound, std::array<T, K> upper_bound) {
      this->lower_bound = lower_bound;
      this->upper_bound = upper_bound;
      this->upper_bound[0] += 1;
      this->state = lower_bound;
    }
    iterator(std::array<T, K> lower_bound, std::array<T, K> upper_bound,
             std::array<T, K> state) {
      this->lower_bound = lower_bound;
      this->upper_bound = upper_bound;
      this->upper_bound[0] += 1;
      this->state = state;
    }
    std::array<T, K> operator++() {
      for (size_t i = K - 1; i >= 0; i--) {
        this->state[i]++;
        if (this->state[i] >= this->upper_bound[i])
          this->state[i] = this->lower_bound[i];
        else
          break;
      }
      return this->state;
    }
    std::array<T, K> operator++(int) {
      const auto res = this->state;
      ++this;
      return res;
    }
    std::array<T, K> operator--() {
      for (size_t i = K - 1; i >= 0; i--) {
        if (this->state[i] > this->lower_bound[i]) {
          this->state[i]--;
          break;
        } else {
          this->state[i] = this->upper_bound[i] - 1;
        }
      }
      return this->state;
    }
    std::array<T, K> operator--(int) {
      const auto res = this->state;
      --this;
      return res;
    }
    std::array<T, K> operator*() { return this->state; };
    pointer operator->() { return &(this->state); };
    bool operator==(const iterator &other) const {
      return this->state == other.state;
    }
    bool operator!=(const iterator &other) const {
      return this->state != other.state;
    }
  };

  iterator begin() { return iterator(this->lower_bound, this->upper_bound); };
  iterator end() {
    std::array<T, K> state;
    state.fill(0);
    state[0] = this->upper_bound[0];
    return iterator(this->lower_bound, this->upper_bound, state);
  };
};

}  // namespace

namespace MSO {
struct BadInitialization : public std::runtime_error {
  BadInitialization(char const *const message) : std::runtime_error(message){};
  BadInitialization(const std::string &message) : std::runtime_error(message){};
};
struct BadArgumentSize : public std::runtime_error {
  BadArgumentSize(char const *const message) : std::runtime_error(message){};
  BadArgumentSize(const std::string &message) : std::runtime_error(message){};
};

template <typename T, typename M = double, size_t N = 3>
/* K is number of dimensions */
class MSO {
 public:
  static const size_t ndim = N;
  typedef std::array<coord_type, N> Point;
  typedef M mu_type;

 private:
  std::vector<int8_t> neighbourhood;
  std::vector<mu_type> distances;
  std::vector<mu_type> mu_array;
  std::vector<T> res_components_array;
  std::vector<bool> sprawl_area_array;
  std::array<coord_type, ndim> size;
  Point lower_bound;
  Point upper_bound;
  std::vector<mu_type> fdt_array;
  T components_num;
  bool use_background = false;
  T *components;
  const T background_component = 1;
  size_t steps = 0;

 public:
  MSO() {
    this->components = nullptr;
    this->size = {0};
  };

  void set_use_background(bool val) { this->use_background = val; }

  void set_components_num(T components_num) {
    this->components_num = components_num;
  }

  void erase_data() {
    /* clean pointers, do not free the memory */
    this->components = nullptr;
    this->size.fill(0);
  }

  inline size_t get_length() const { return calculate_area_size(this->size); }

  inline std::array<size_t, ndim> dimension_size() const {
    return calculate_dimension_size(this->size);
  }

  template <typename W>
  void set_data(T *components, W size) {
    this->components = components;
    for (size_t i = 0; i < ndim; i++) {
      this->size[i] = size[i];
      this->upper_bound[i] = size[i];
      this->lower_bound[i] = 0;
    }
    if (this->get_length() != this->mu_array.size()) this->mu_array.clear();
    this->steps = 0;
  }

  template <typename W>
  void set_bounding_box(W lower_bound, W upper_bound) {
    for (size_t i = 0; i < ndim; i++) {
      this->lower_bound[i] = lower_bound[i];
      this->upper_bound[i] = upper_bound[i];
    }
  }

  void set_mu_copy(const std::vector<mu_type> &mu) {
    if (mu.size() != this->get_length())
      throw std::length_error(
          "Size of mu array (" + std::to_string(mu.size()) +
          ") need to be equal to size of components array (" +
          std::to_string(this->get_length()) + ")");
    this->mu_array = mu;
    this->steps = 0;
  }
  void set_mu_copy(mu_type *mu, size_t length) {
    if (length != this->get_length())
      throw std::length_error(
          "Size of mu array (" + std::to_string(length) +
          ") need to be equal to size of components array (" +
          std::to_string(this->get_length()) + ")");
    this->mu_array = std::vector<mu_type>(mu, mu + length);
    this->steps = 0;
  }

  void set_mu_swap(std::vector<mu_type> &mu) {
    if (mu.size() != this->get_length())
      throw std::length_error(
          "Size of mu array (" + std::to_string(mu.size()) +
          ") need to be equal to size of components array (" +
          std::to_string(this->get_length()) + ")");
    this->mu_array.swap(mu);
    this->steps = 0;
  }

  void set_neighbourhood(std::vector<int8_t> neighbourhood,
                         std::vector<mu_type> distances) {
    if (neighbourhood.size() != ndim * distances.size()) {
      throw std::length_error(
          "Size of neighbouthood need to be 3* Size of distances");
    }
    this->neighbourhood = neighbourhood;
    this->distances = distances;
    this->steps = 0;
  }

  void set_neighbourhood(int8_t *neighbourhood, mu_type *distances,
                         size_t neigh_size) {
    this->neighbourhood =
        std::vector<int8_t>(neighbourhood, neighbourhood + 3 * neigh_size);
    this->distances = std::vector<double>(distances, distances + neigh_size);
    this->steps = 0;
  }

  void compute_FDT(std::vector<mu_type> &array) const {
    if (this->get_length() == 0)
      throw BadInitialization(
          "call FDT calculation before set coordinates data");
    if (this->mu_array.size() == 0)
      throw BadInitialization("call FDT calculation before set mu array");
    if (this->mu_array.size() != this->get_length())
      throw BadInitialization(
          "call FDT calculation with mu_array of different size (" +
          std::to_string(this->mu_array.size()) + ") than image size (" +
          std::to_string(this->get_length()) + ")");
    if (array.size() !=
        calculate_area_size(this->upper_bound - this->lower_bound))
      throw BadArgumentSize(
          "call FDT calculation with array of different size (" +
          std::to_string(array.size()) + ") than expected (" +
          std::to_string(
              calculate_area_size(this->upper_bound - this->lower_bound)) +
          ")");

    const std::array<size_t, ndim> dimension_size =
        calculate_dimension_size(this->upper_bound - this->lower_bound);
    const std::array<size_t, ndim> global_dimension_size =
        this->dimension_size();
    Point coord, coord2;
    size_t position, neigh_position, array_position, array_neigh_position;
    my_queue<Point> queue;
    double val, mu_value, fdt_value;
    std::vector<bool> coord_in_queue(this->get_length(), false);
    // std::cout << "Neighbourhood: " << this->neighbourhood << std::endl <<
    // "Distances: " << this->distances << std::endl;
    auto bounds =
        ArrayLimits<coord_type, ndim>(this->lower_bound, this->upper_bound);
    for (auto coord : bounds) {
      position = calculate_position(coord, global_dimension_size);
      array_position =
          calculate_position(coord - this->lower_bound, global_dimension_size);
      array[array_position] = std::numeric_limits<mu_type>::max();
      if (this->components[position] == this->background_component) {
        array[position] = 0;
        for (size_t i = 0; i < 3 * this->distances.size(); i += 3) {
          for (size_t j = 0; j < ndim; j++)
            coord2[j] = coord[j] + this->neighbourhood[i + j];
          if (outside_bounds(coord2, lower_bound, upper_bound)) {
            continue;
          }
          if (components[calculate_position(coord2, global_dimension_size)] ==
              0) {
            queue.push(coord);
            coord_in_queue[calculate_position(coord, global_dimension_size)] =
                true;
            break;
          }
        }
      }
    }
    // std::cout << std::endl << "Queue size " << queue.get_size() << std::endl;
    size_t count = 0;
    while (!queue.empty()) {
      count += 1;
      coord = queue.front();
      queue.pop();
      position = calculate_position(coord, global_dimension_size);
      array_position =
          calculate_position(coord - this->lower_bound, dimension_size);
      mu_value = this->mu_array[position];
      fdt_value = array[array_position];
      for (size_t i = 0; i < this->distances.size(); i++) {
        for (size_t j = 0; j < ndim; j++)
          coord2[j] = coord[j] + this->neighbourhood[3 * i + j];
        if (outside_bounds(coord2, lower_bound, upper_bound)) continue;
        neigh_position = calculate_position(coord2, global_dimension_size);
        array_neigh_position =
            calculate_position(coord2 - this->lower_bound, dimension_size);
        if (this->components[neigh_position] != 0) continue;
        val = (this->mu_array[neigh_position] + mu_value) * this->distances[i] /
              2;
        if (array[array_neigh_position] > val + fdt_value) {
          array[array_neigh_position] = val + fdt_value;
          if (!coord_in_queue[neigh_position]) {
            coord_in_queue[neigh_position] = true;
            queue.push(coord2);
          }
        }
      }
      coord_in_queue[position] = false;
    }
    // std::cout << "Count " << count << std::endl;
  };

  size_t optimum_erosion_calculate(const std::vector<mu_type> &fdt_array,
                                   std::vector<T> &components_arr,
                                   std::vector<bool> &sprawl_area) {
    Point coord, coord2;
    size_t position, neigh_position;
    mu_type val, val2;
    std::vector<mu_type> distances_from_components(fdt_array.size(), 0);
    std::vector<my_queue<Point>> queues(this->components_num + 1);
    std::vector<bool> coord_in_queue(this->get_length(), false);
    auto bounds =
        ArrayLimits<coord_type, ndim>(this->upper_bound - this->lower_bound);
    const std::array<size_t, ndim> dimension_size =
        calculate_dimension_size(this->upper_bound - this->lower_bound);
    const size_t area_size = bounds.size();

    for (auto coord : bounds) {
      position = calculate_position(coord, dimension_size);
      if (components_arr[position] != 0) {
        distances_from_components[position] = fdt_array[position];
        for (size_t i = 0; i < 3 * this->distances.size(); i += 3) {
          for (size_t j = 0; j < ndim; j++)
            coord2[j] = coord[j] + this->neighbourhood[i + j];
          if (outside_bounds(coord2, lower_bound, upper_bound)) continue;
          if (sprawl_area[calculate_position(coord2, dimension_size)] == true) {
            queues[components_arr[position]].push(coord);
            coord_in_queue[calculate_position(coord, dimension_size)] = true;
            break;
          }
        }
      }
    }
    // size_t k = 0;
    for (auto &queue : queues) {
      // std::cerr << "Queue " << k << " size " << queue.get_size() <<
      // std::endl;
      size_t count_steps = 0;
      while (!queue.empty()) {
        coord = queue.front();
        /*if (k==3){
          std::cerr << "Coord " << coord << std::endl;
        }*/
        queue.pop();
        position = calculate_position(coord, dimension_size);
        val = distances_from_components[position];
        for (size_t i = 0; i < this->distances.size(); i++) {
          for (size_t j = 0; j < ndim; j++)
            coord2[j] = coord[j] + this->neighbourhood[3 * i + j];
          if (outside_bounds(coord2, lower_bound, upper_bound)) continue;
          neigh_position = calculate_position(coord2, dimension_size);
          if (sprawl_area[neigh_position] == false) continue;
          val2 = std::min(val, fdt_array[neigh_position]);
          if (val2 < distances_from_components[neigh_position] -
                         std::numeric_limits<mu_type>::epsilon())
            continue;
          if ((fabs(val2 - distances_from_components[neigh_position])) <
                  std::numeric_limits<mu_type>::epsilon() &&
              ((components_arr[neigh_position] == components_arr[position]) ||
               (components_arr[neigh_position] ==
                std::numeric_limits<T>::max())))
            continue;
          if (val2 > distances_from_components[neigh_position] +
                         std::numeric_limits<mu_type>::epsilon()) {
            distances_from_components[neigh_position] = val2;
            components_arr[neigh_position] = components_arr[position];
          } else {
            components_arr[neigh_position] = std::numeric_limits<T>::max();
          }
          if (!coord_in_queue[neigh_position]) {
            coord_in_queue[neigh_position] = true;
            queue.push(coord2);
            count_steps++;
          }
        }
        if (count_steps > 3 * area_size){
          throw std::runtime_error("two many steps: constrained dilation");
        }
        coord_in_queue[position] = false;
      }
      // k++;
    }
    size_t count = 0;
    for (auto &el : components_arr) {
      if (el == std::numeric_limits<T>::max()) {
        el = 0;
      }
    }
    for (size_t i = 0; i < bounds.size(); i++) {
      if (sprawl_area[i] && components_arr[i] > 0) {
        sprawl_area[i] = false;
        count++;
      }
    }
    return count;
  };

  size_t constrained_dilation(const std::vector<mu_type> &fdt_array,
                              std::vector<T> &components_arr,
                              std::vector<bool> &sprawl_area) const {
    if (this->get_length() == 0) throw BadInitialization("Zero sized array");
    if (this->mu_array.size() != this->get_length())
      throw BadInitialization("mu array size (" +
                              std::to_string(this->mu_array.size()) +
                              ") do not fit to size of data (" +
                              std::to_string(this->get_length()) + ")");
    Point coord, coord2;
    size_t position, neigh_position, position_global;
    mu_type val, val2, dist_val;
    std::vector<mu_type> distances_from_components(
        fdt_array.size(), std::numeric_limits<mu_type>::max());
    std::vector<my_queue<Point>> queues(this->components_num + 1);
    my_queue<Point> queue_copy;
    std::vector<bool> coord_in_queue(this->get_length(), false);
    auto bounds =
        ArrayLimits<coord_type, ndim>(this->upper_bound - this->lower_bound);
    std::vector<T> morphological_neighbourhood = components_arr;
    const std::array<size_t, ndim> dimension_size =
        calculate_dimension_size(this->upper_bound - this->lower_bound);
    const std::array<size_t, ndim> global_dimension_size =
        this->dimension_size();

    const size_t area_size = bounds.size();

    // Put borders of components to queue
    for (auto coord : bounds) {
      position = calculate_position(coord, dimension_size);
      if (components_arr[position] != 0) {
        for (size_t i = 0; i < 3 * this->distances.size(); i += 3) {
          for (size_t j = 0; j < ndim; j++)
            coord2[j] = coord[j] + this->neighbourhood[i + j];
          if (outside_bounds(coord2, lower_bound, upper_bound)) continue;
          if (sprawl_area[calculate_position(coord2, dimension_size)] == true) {
            queues[components_arr[position]].push(coord);
            distances_from_components[position] =
                0;  // this->mu_array[calculate_position(coord +
                    // this->lower_bound, global_dimension_size)];
            coord_in_queue[position] = true;
            break;
          }
        }
      }
    }
    // std::cerr << "start dilation\n";
    T comp_num = 0;
    for (auto &queue : queues) {
      //std::cerr << "Queue " << (int) comp_num << " size " << queue.get_size() <<
      //std::endl;
      queue_copy = queue;
      // std::cerr << "# Queue1 " << queue.get_size() << " Queue2 " <<
      // queue_copy.get_size() << std::endl;
      // Calculate area which can be reached by monoticall path (firs part of
      // Morphological neighborhood)
      while (!queue.empty()) {
        coord = queue.front();
        queue.pop();
        position = calculate_position(coord, dimension_size);
        val = fdt_array[position];
        for (size_t i = 0; i < this->distances.size(); i++) {
          for (size_t j = 0; j < ndim; j++)
            coord2[j] = coord[j] + this->neighbourhood[3 * i + j];
          if (outside_bounds(coord2, lower_bound, upper_bound)) continue;
          neigh_position = calculate_position(coord2, dimension_size);
          if (sprawl_area[neigh_position] == false) continue;
          if (fdt_array[neigh_position] <= val &&
              morphological_neighbourhood[neigh_position] != comp_num &&
              morphological_neighbourhood[neigh_position] !=
                  std::numeric_limits<T>::max()) {
            morphological_neighbourhood[neigh_position] = comp_num;
            queue.push(coord2);
          }
        }
      }
      //std::cerr << "start constrainde dilation step " << (int) comp_num << std::endl;
      // std::cerr << "Queue1 " << queue.get_size() << " Queue2 " <<
      // queue_copy.get_size() << std::endl;
      // calculate constrained dilation
      size_t count_steps = 0;
      while (!queue_copy.empty()) {
        coord = queue_copy.front();
        queue_copy.pop();
        position = calculate_position(coord, dimension_size);
        dist_val = distances_from_components[position];
        position_global = calculate_position(coord + this->lower_bound,
                                             global_dimension_size);
        val = this->mu_array[position_global];
        // std::cerr << "Coord " << coord <<  " val " << val << std::endl;
        for (size_t i = 0; i < this->distances.size(); i++) {
          for (size_t j = 0; j < ndim; j++)
            coord2[j] = coord[j] + this->neighbourhood[3 * i + j];
          if (outside_bounds(coord2, lower_bound, upper_bound)) continue;
          neigh_position = calculate_position(coord2, dimension_size);
          if (sprawl_area[neigh_position] == false ||
              morphological_neighbourhood[neigh_position] != comp_num)
            continue;
          val2 = (this->mu_array[calculate_position(coord + this->lower_bound,
                                                    global_dimension_size)] +
                  val) *
                 this->distances[i] / 2;
          val2 = dist_val + val2;
          if (val2 + std::numeric_limits<mu_type>::epsilon() >= fdt_array[neigh_position]) {
            // std::cerr << "    coord(fdt) " << coord2 << " " << val2 << " - "
            // << fdt_array[neigh_position] << std::endl;
            continue;
          }
          if (distances_from_components[neigh_position] < val2 + std::numeric_limits<mu_type>::epsilon()) {
            // std::cerr << "    coord(dist) " << coord2 << " " << val2 << " - "
            // << distances_from_components[neigh_position] << std::endl;
            continue;
          }
          if (fabs(val2 - fdt_array[neigh_position]) < std::numeric_limits<mu_type>::epsilon()) {
            if (components_arr[neigh_position] == std::numeric_limits<T>::max())
                continue;
            else
                components_arr[neigh_position] = std::numeric_limits<T>::max();
          }
          if (val2 < fdt_array[neigh_position]) {
            components_arr[neigh_position] = components_arr[position];
            distances_from_components[neigh_position] = val2;
          }
          if (!coord_in_queue[neigh_position]) {
            count_steps++;
            queue_copy.push(coord2);
            coord_in_queue[neigh_position] = true;
          }
        }
        if (count_steps > 3 * area_size){
          throw std::runtime_error("two many steps: constrained dilation");
        }


        coord_in_queue[position] = false;
        // distances[position] = this->mu_array[calculate_position(coord +
        // this->lower_bound, global_dimension_size)]
      }
      // std::cerr << "end constrainde dilation step " << (int) comp_num << std::endl;
      // std::cerr << " component change " << count3 << " ";
      // std::cerr << "Queue1 " << queue.get_size() << " Queue2 " <<
      // queue_copy.get_size() << std::endl;
      comp_num++;
    }
    size_t count = 0;
    for (auto &el : components_arr) {
      if (el == std::numeric_limits<T>::max()) {
        el = 0;
      }
    }
    for (size_t i = 0; i < bounds.size(); i++) {
      if (sprawl_area[i] && components_arr[i] > 0) {
        sprawl_area[i] = false;
        count++;
      }
    }
    return count;
  };

  std::vector<T> cut_components() {
    auto bounds =
        ArrayLimits<coord_type, ndim>(this->lower_bound, this->upper_bound);
    std::vector<T> res(bounds.size());
    T val;
    const std::array<size_t, ndim> dimension_size =
        calculate_dimension_size(this->upper_bound - this->lower_bound);
    const std::array<size_t, ndim> global_dimension_size =
        this->dimension_size();
    if (this->use_background) {
      for (auto coord : bounds) {
        res[calculate_position(coord, dimension_size)] =
            this->components[calculate_position(coord, global_dimension_size)];
      }
    } else {
      for (auto coord : bounds) {
        val =
            this->components[calculate_position(coord, global_dimension_size)];
        if (val == 1) val = 0;
        res[calculate_position(coord, dimension_size)] = val;
      }
    }
    return res;
  }

  std::vector<bool> get_sprawl_area() {
    auto bounds =
        ArrayLimits<coord_type, ndim>(this->lower_bound, this->upper_bound);
    std::vector<bool> res(bounds.size());
    const std::array<size_t, ndim> dimension_size =
        calculate_dimension_size(this->upper_bound - this->lower_bound);
    const std::array<size_t, ndim> global_dimension_size =
        this->dimension_size();
    for (auto coord : bounds) {
      res[calculate_position(coord, dimension_size)] =
          this->components[calculate_position(coord, global_dimension_size)] ==
          0;
    }

    return res;
  }

  size_t run_MSO(size_t steps_limits = 1) {
    if (this->components_num == 0)
      throw BadInitialization("Wrong number of components seted");
    size_t total_changes = 0;
    if (steps_limits == 0) steps_limits = 1;
    if (steps_limits < this->steps) this->steps = 0;
    size_t count_changes = 1;
    if (this->steps == 0) {
      this->fdt_array.resize(
          calculate_area_size(this->upper_bound - this->lower_bound));
      std::fill(this->fdt_array.begin(), this->fdt_array.end(), 0);
      this->compute_FDT(this->fdt_array);
      this->res_components_array = this->cut_components();
      this->sprawl_area_array = this->get_sprawl_area();
    }
    while (this->steps < steps_limits && count_changes > 0) {
      //std::cerr << "loop " << count_changes << std::endl;
      count_changes = 0;
      count_changes += optimum_erosion_calculate(
          this->fdt_array, this->res_components_array, this->sprawl_area_array);
      //std::cerr << "loop2\n";
      count_changes += constrained_dilation(
          this->fdt_array, this->res_components_array, this->sprawl_area_array);
      total_changes += count_changes;
      //std::cerr << "loop3\n";
      this->steps++;
    }
    //std::cerr << "end\n";
    if (count_changes == 0) {
      this->steps--;
    }
    return total_changes;
  }

  size_t steps_done() { return this->steps; }

  std::vector<T> get_result_catted() const {
    return this->res_components_array;
  }

  std::vector<mu_type> get_fdt() { return this->fdt_array; }
};

template <typename R>
void inline shrink(R &val) {
  if (val > 1)
    val = 1;
  else if (val < 0)
    val = 0;
}

template <typename R, typename T>
class MuCalc {
 public:
  static std::vector<R> calculate_mu_array(T *array, size_t length,
                                           T lower_bound, T upper_bound) {
    std::vector<R> result(length, 0);
    R mu;
    for (size_t i = 0; i < length; i++) {
      mu = (R)(array[i] - lower_bound) / (upper_bound - lower_bound);
      shrink(mu);
      result[i] = mu;
    }
    return result;
  }

  static std::vector<R> calculate_reflection_mu_array(T *array, size_t length,
                                                      T lower_bound,
                                                      T upper_bound) {
    std::vector<R> result(length, 0);
    R mu;
    for (size_t i = 0; i < length; i++) {
      mu = (R)(array[i] - lower_bound) / (upper_bound - lower_bound);
      shrink(mu);
      if (mu < 0.5) mu = 1 - mu;
      result[i] = mu;
    }
    return result;
  }

  static std::vector<R> calculate_two_object_mu(T *array, size_t length,
                                                T lower_bound, T upper_bound,
                                                T lower_mid_bound,
                                                T upper_mid_bound) {
    std::vector<R> result(length, 0);
    R mu;
    T pixel_val;
    for (size_t i = 0; i < length; i++) {
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

  static std::vector<R> calculate_mu_array_masked(T *array, size_t length,
                                                  T lower_bound, T upper_bound,
                                                  uint8_t *mask) {
    std::vector<R> result(length, 0);
    R mu;
    for (size_t i = 0; i < length; i++) {
      if (mask[i] == 0) continue;
      mu = (R)(array[i] - lower_bound) / (upper_bound - lower_bound);
      shrink(mu);
      result[i] = mu;
    }
    return result;
  }

  static std::vector<R> calculate_reflection_mu_array_masked(
      T *array, size_t length, T lower_bound, T upper_bound, uint8_t *mask) {
    std::vector<R> result(length, 0);
    R mu;
    for (size_t i = 0; i < length; i++) {
      if (mask[i] == 0) continue;
      mu = (R)(array[i] - lower_bound) / (upper_bound - lower_bound);
      shrink(mu);
      if (mu < 0.5) mu = 1 - mu;
      result[i] = mu;
    }
    return result;
  }
  static std::vector<R> calculate_two_object_mu_masked(
      T *array, size_t length, T lower_bound, T upper_bound, T lower_mid_bound,
      T upper_mid_bound, uint8_t *mask) {
    std::vector<R> result(length, 0);
    R mu;
    T pixel_val;
    for (size_t i = 0; i < length; i++) {
      if (mask[i] == 0) continue;
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
}  // namespace MSO
