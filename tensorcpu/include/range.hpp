//
// Created by tejag on 2024-04-11.
//

#ifndef TENSORC_RANGE_HPP
#define TENSORC_RANGE_HPP

#include <iterator>
#include <memory>

class Range {
public:
    using iterator_category = std::random_access_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using size_type = std::size_t;
    using value_type = int64_t;
    using pointer = value_type *;
    using reference = value_type &;

    int64_t index;

    explicit Range(int64_t index) : index(index) {};

    Range(const Range &other) = default;

    reference operator*() { return index; }

    pointer operator->() { return &index; }

    Range &operator++() {
        index += 1;
        return *this;
    }

    Range operator++(int) &{
        Range tmp = *this;
        ++(*this);
        return tmp;
    }

    Range &operator--() {
        index -= 1;
        return *this;
    }

    Range operator--(int) &{
        Range tmp = *this;
        --(*this);
        return tmp;
    }

    Range &operator+=(difference_type n) {
        index += n;
        return *this;
    }

    Range &operator-=(difference_type n) {
        index -= n;
        return *this;
    }

    Range operator+(difference_type n) const {
        Range tmp = *this;
        return tmp += n;
    }

    Range operator-(difference_type n) const {
        Range tmp = *this;
        return tmp -= n;
    }

    difference_type operator-(const Range &rhs) const {
        difference_type ret = index - rhs.index;
        return ret;
    }

    friend bool operator==(const Range &a, const Range &b) {
        return a.index == b.index;
    };

    friend bool operator!=(const Range &a, const Range &b) {
        return a.index != b.index;
    };

    friend bool operator<(const Range &a, const Range &b) {
        return a.index < b.index;
    };

    friend bool operator>(const Range &a, const Range &b) {
        return a.index > b.index;
    };

    friend bool operator<=(const Range &a, const Range &b) {
        return a.index <= b.index;
    };

    friend bool operator>=(const Range &a, const Range &b) {
        return a.index >= b.index;
    };
};

#endif //TENSORC_RANGE_HPP
