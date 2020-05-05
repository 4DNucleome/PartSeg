#include <queue>
#include <vector>
#include <cstddef>
#include <utility>
#include <iostream>

static const size_t buffer_size = 5000;

template<typename T>
class my_queue {
public:
    my_queue() : in_vector(buffer_size), out_vector(buffer_size) {}

    T front() const{
        if (this->single_vector)
            return this->in_vector[this->out_position];
        return this->out_vector[this->out_position];
    }

    void pop(){
        this->size--;
        this->out_position++;
        if (this->single_vector){
            return;
        }
        if (this->out_position == buffer_size){
            if (this->in_queue.empty()){
                this->single_vector = true;
            } else {
                this->out_vector = this->in_queue.front();
                this->in_queue.pop();
            }
            this->out_position = 0;
        }
    }

    void push(T& v){
        this->in_vector[this->in_position] = v;
        this->in_position++;

        if (this->in_position == buffer_size){
            if (this->single_vector){
                std::swap(this->out_vector, this->in_vector);
                this->single_vector = false;
            } else {
                this->in_queue.push(this->in_vector);
            }
            this->in_position = 0;
        }
        this->size++;

    }
    bool empty() const{
        return this->size == 0;
    }

    size_t get_size() const {
        return this->size;
    }

protected:
    std::queue<std::vector<T> > in_queue;
    std::vector<T> in_vector;
    std::vector<T> out_vector;
    std::size_t in_position = 0;
    std::size_t out_position = 0;
    std::size_t size = 0;
    bool single_vector = true;
};
