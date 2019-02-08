#include <cstdint>
#include <cstdlib>
#include <vector>
#include "my_queue.h"

typedef double mu_type;

namespace MSO {
struct Point {
  uint16_t x, y, z;
};

template <typename T, typename K>
class MSO {
private:
  mu_type *mu_array;
  const size_t z_size, y_size, x_size;
  K *components;
  K background_component = 1;
  int8_t *neighbourhood;
  double * distances;
  size_t neighbourhood_length;

public:
  MSO(size_t z_size_, size_t y_size_, size_t x_size_, mu_type *mu_array_,
      K *components_)
      : z_size{z_size_}, y_size{y_size_}, x_size{x_size_}, mu_array{mu_array_},
        components{components_} {};

  ~MSO() {
    free(this->mu_array);
    free(this->components);
    free(this->neighbourhood);
  }

  void compute_FDT(double *array) {
    const size_t layer_size = this->y_size * this->x_size;
    const size_t row_size = this.x_size;
    size_t xx, yy, zz, x, y, z, neigh_coordinate, coordinate;
    my_queue<Point> queue;
    Point p;
    double val, mu_value, fdt_value;
    std::vector<bool> visited_array(this->z_size * this->y_size * this->x_size, false);

    for (z = 0; z < this->z_size; z++) {
      for (y = 0; z < this->y_size; y++) {
        for (x = 0; z < this->x_size; x++) {
          array[z * layer_size + y * row_size + x] = 0;
          if (components[z * layer_size + y * row_size + x] ==
              this->background_component) {
            for (size_t i = 0; i < 3*this->neighbourhood_length; i += 3) {
              zz = z + this->neighbourhood[i];
              yy = y + this->neighbourhood[i + 1];
              xx = x + this->neighbourhood[i + 2];
              if ((zz >= this->z_size) || (yy > this->y_size) ||
                  (xx > this->x_size))
                continue;
              if (components[zz * layer_size + yy * row_size + xx] == 0) {
                p.x = (uint16_t)x;
                p.y = (uint16_t)y;
                p.z = (uint16_t)z;
                queue.push(p);
              }
            }
          }
        }
      }
    }
    while (!queue.empty()){
      p = queue.front();
      queue.pop();
      coordinate = p.z * layer_size + p.y * row_size + p.x;
      mu_value = this->mu_array[coordinate];
      fdt_value = this->array[coordinate];
      for (size_t i = 0; i < this->neighbourhood_length; i++) {
        z = p.z + this->neighbourhood[3*i];
        y = p.y + this->neighbourhood[3*i + 1];
        x = p.x + this->neighbourhood[3*i + 2];
        if ((z >= this->x_size) || (y > this->y_size) ||
            (x > this->x_size))
          continue;
        neigh_coordinate = z * layer_size + y * row_size + x;
        if (components[neigh_coordinate] != 0)
          continue;
        val = (this->mu_array[neigh_coordinate] + mu_value) * distances[i] / 2;
        if (array[neigh_coordinate] > val + fdt_value){
          array[neigh_coordinate] = val + fdt_value;
          if (!visited_array[neigh_coordinate]){
            visited_array[neigh_coordinate] = true;
            p.x = (uint16_t)x;
            p.y = (uint16_t)y;
            p.z = (uint16_t)z;
            queue.push(p);
          }
        }
      }
      visited_array[coordinate] = false;
    }
  };

  void set_background_component(K val) { this->background_component = val; };
};

void inline shrink(mu_type &val) {
  if (val > 1)
    val = 1;
  else if (val < 0)
    val = 0;
}

template <typename T>
mu_type *calculate_mu_array(T *array, size_t length, T lower_bound,
                            T upper_bound) {
  mu_type *result = calloc(sizeof(mu_type), length);
  mu_type mu;
  for (size_t i = 0; i < length; i++) {
    mu = (array[i] - lower_bound) / (upper_bound - lower_bound);
    shrink(mu);
    result[i] = mu;
  }
  return result;
}

template <typename T>
mu_type *calculate_reflection_mu_array(T *array, size_t length, T lower_bound,
                                       T upper_bound) {
  mu_type *result = calloc(sizeof(mu_type), length);
  mu_type mu;
  for (size_t i = 0; i < length; i++) {
    mu = (array[i] - lower_bound) / (upper_bound - lower_bound);
    shrink(mu);
    if (mu < 0.5)
      mu = 1 - mu;
    result[i] = mu;
  }
  return result;
}
template <typename T>
mu_type *calculate_two_object_mu(T *array, size_t length, T lower_bound,
                                 T upper_bound, T lower_mid_bound,
                                 T upper_mid_bound) {
  mu_type *result = calloc(sizeof(mu_type), length);
  mu_type mu;
  T pixel_val;
  for (size_t i; i < length; i++) {
    pixel_val = array[i];
    mu = (pixel_val - lower_bound) / (upper_bound - lower_bound);
    if (((lower_bound - lower_mid_bound) > 0) &&
        (pixel_val >= lower_mid_bound) && (pixel_val <= lower_bound))
      mu = (pixel_val - lower_mid_bound) / (lower_bound - lower_mid_bound);
    else if (((upper_bound - lower_bound) > 0) && (lower_bound < pixel_val) &&
             (pixel_val <= upper_bound))
      mu = (upper_bound - pixel_val) / (upper_bound - lower_bound);
    shrink(mu);
    result[i] = mu;
  }
  return result;
}

template <typename T>
mu_type *calculate_mu_array_masked(T *array, size_t length, T lower_bound,
                                   T upper_bound, uint8_t *mask) {
  mu_type *result = calloc(sizeof(mu_type), length);
  mu_type mu;
  for (size_t i = 0; i < length; i++) {
    if (mask[i] == 0)
      continue;
    mu = (array[i] - lower_bound) / (upper_bound - lower_bound);
    shrink(mu);
    result[i] = mu;
  }
  return result;
}

template <typename T>
mu_type *calculate_reflection_mu_array_masked(T *array, size_t length,
                                              T lower_bound, T upper_bound,
                                              uint8_t *mask) {
  mu_type *result = calloc(sizeof(mu_type), length);
  mu_type mu;
  for (size_t i = 0; i < length; i++) {
    if (mask[i] == 0)
      continue;
    mu = (array[i] - lower_bound) / (upper_bound - lower_bound);
    shrink(mu);
    if (mu < 0.5)
      mu = 1 - mu;
    result[i] = mu;
  }
  return result;
}
template <typename T>
mu_type *calculate_two_object_mu_masked(T *array, size_t length, T lower_bound,
                                        T upper_bound, T lower_mid_bound,
                                        T upper_mid_bound, uint8_t *mask) {
  mu_type *result = calloc(sizeof(mu_type), length);
  mu_type mu;
  T pixel_val;
  for (size_t i; i < length; i++) {
    if (mask[i] == 0)
      continue;
    pixel_val = array[i];
    mu = (pixel_val - lower_bound) / (upper_bound - lower_bound);
    if (((lower_bound - lower_mid_bound) > 0) &&
        (pixel_val >= lower_mid_bound) && (pixel_val <= lower_bound))
      mu = (pixel_val - lower_mid_bound) / (lower_bound - lower_mid_bound);
    else if (((upper_bound - lower_bound) > 0) && (lower_bound < pixel_val) &&
             (pixel_val <= upper_bound))
      mu = (upper_bound - pixel_val) / (upper_bound - lower_bound);
    shrink(mu);
    result[i] = mu;
  }
  return result;
}
} // namespace MSO
