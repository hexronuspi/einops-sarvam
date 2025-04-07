#define EIGEN_STACK_ALLOCATION_LIMIT 10000000
#define EIGEN_USE_THREADS
#define EIGEN_DONT_PARALLELIZE

#include <unsupported/Eigen/CXX11/Tensor>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <string>
#include <map>
#include <set>
#include <sstream>
#include <regex>
#include <stdexcept>
#include <algorithm>
#include <iterator>

namespace py = pybind11;
using Index = Eigen::Index;
using TensorFixed = Eigen::Tensor<float, 10, Eigen::RowMajor>;

class ParserError : public std::runtime_error {
public:
    explicit ParserError(const std::string& message) : std::runtime_error(message) {}
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct AxisSpec {
    std::string name;
    bool is_group;
    bool is_ellipsis;
    AxisSpec(const std::string& n, bool g, bool e) : name(n), is_group(g), is_ellipsis(e) {}
};

class PatternParser {
public:
    static std::pair<std::vector<AxisSpec>, std::vector<AxisSpec>> parse(
        const std::string& input_spec_str, const std::string& output_spec_str) {
        std::vector<AxisSpec> input_spec = parse_spec(input_spec_str);
        std::vector<AxisSpec> output_spec = parse_spec(output_spec_str);

        bool input_has_ellipsis = std::any_of(input_spec.begin(), input_spec.end(),
                                               [](const AxisSpec &s){ return s.is_ellipsis; });
        bool output_has_ellipsis = std::any_of(output_spec.begin(), output_spec.end(),
                                                [](const AxisSpec &s){ return s.is_ellipsis; });
        if (input_has_ellipsis != output_has_ellipsis) {
            throw ParserError("Ellipsis must be present in both input and output or neither");
        }

        std::set<std::string> input_axes, output_axes;
        for (const auto& item : input_spec) {
            if (!item.is_ellipsis && item.name != "1" && !item.is_group)
                input_axes.insert(item.name);
        }
        for (const auto& item : output_spec) {
            if (!item.is_ellipsis && !item.is_group)
                output_axes.insert(item.name);
        }

        std::set<std::string> missing = output_axes;
        for (const auto& ax : input_axes)
            missing.erase(ax);
        if (!missing.empty()) {
            bool has_one = std::any_of(input_spec.begin(), input_spec.end(),
                                       [](const AxisSpec &s){ return s.name == "1"; });
            if (!has_one) {
                throw ParserError("Output axes not in input without '1': " + set_to_string(missing));
            }
        }

        return std::make_pair(input_spec, output_spec);
    }

private:
    static std::vector<AxisSpec> parse_spec(const std::string& spec_str) {
        std::regex token_re(R"(\w+|->|\.\.\.|\(|\)|1)");
        std::sregex_iterator it(spec_str.begin(), spec_str.end(), token_re);
        std::sregex_iterator end;
        std::vector<std::string> tokens;
        for (; it != end; ++it) {
            tokens.push_back(it->str());
        }
        if (tokens.empty())
            throw ParserError("Empty pattern string");

        std::vector<AxisSpec> spec;
        std::vector<std::string> group;
        bool in_group = false;

        for (size_t i = 0; i < tokens.size(); ++i) {
            const std::string& token = tokens[i];
            if (token == "(") {
                if (in_group)
                    throw ParserError("Nested parentheses not supported");
                in_group = true;
                group.clear();
            } else if (token == ")") {
                if (!in_group)
                    throw ParserError("Unmatched closing parenthesis");
                in_group = false;
                for (const auto& ax : group)
                    spec.push_back(AxisSpec(ax, true, false));
            } else if (token == "->") {
                throw ParserError("Unexpected '->' in specification");
            } else if (token == "...") {
                if (std::any_of(spec.begin(), spec.end(),
                                [](const AxisSpec &s){ return s.is_ellipsis; }))
                    throw ParserError("Multiple ellipses not supported");
                if (in_group)
                    group.push_back(token);
                else
                    spec.push_back(AxisSpec("...", false, true));
            } else if (std::regex_match(token, std::regex("\\w+|1"))) {
                if (in_group)
                    group.push_back(token);
                else
                    spec.push_back(AxisSpec(token, false, false));
            } else {
                throw ParserError("Invalid token: " + token);
            }
        }
        if (in_group)
            throw ParserError("Unclosed parenthesis");

        return spec;
    }

    static std::string set_to_string(const std::set<std::string>& s) {
        std::ostringstream oss;
        for (const auto& item : s)
            oss << item << " ";
        return oss.str();
    }
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

TensorFixed rearrange(py::array_t<float> tensor, std::string input_spec_str,
                      std::string output_spec_str, py::dict axes_lengths_dict) {
    std::map<std::string, Index> axes_lengths;
    for (auto item : axes_lengths_dict) {
        axes_lengths[item.first.cast<std::string>()] = item.second.cast<Index>();
    }

    auto tensor_info = tensor.request();
    std::vector<Index> in_shape(tensor_info.shape.begin(), tensor_info.shape.end());
    if (in_shape.size() > 10)
        throw std::runtime_error("Tensor rank exceeds maximum supported dimensions (10)");

    Eigen::array<Index, 10> dims;
    std::fill(dims.begin(), dims.end(), 1);
    for (size_t i = 0; i < in_shape.size(); ++i)
        dims[i] = in_shape[i];

    Eigen::TensorMap<TensorFixed> tensor_map(static_cast<float*>(tensor_info.ptr), dims);

    auto specs = PatternParser::parse(input_spec_str, output_spec_str);
    std::vector<AxisSpec>& input_spec = specs.first;
    std::vector<AxisSpec>& output_spec = specs.second;

    std::map<std::string, Index> axis_sizes = axes_lengths;
    size_t shape_pos = 0;
    int ellipsis_dims = 0;
    for (const auto& item : input_spec) {
        if (item.is_ellipsis) {
            ellipsis_dims = static_cast<int>(in_shape.size() - (input_spec.size() - 1));
            for (int j = 0; j < ellipsis_dims; ++j) {
                axis_sizes["batch_" + std::to_string(j)] = in_shape[shape_pos++];
            }
        } else if (item.name == "1") {
            if (shape_pos >= in_shape.size() || in_shape[shape_pos] != 1)
                throw std::runtime_error("Expected singleton axis at position " + std::to_string(shape_pos));
            shape_pos++;
        } else if (!item.is_group) {
            if (shape_pos >= in_shape.size())
                throw std::runtime_error("Input tensor has too few dimensions");
            axis_sizes[item.name] = in_shape[shape_pos++];
        } else {
            shape_pos++;
        }
    }

    std::vector<Index> intermediate_shape;
    std::vector<std::string> intermediate_axes;
    std::vector<std::pair<int, Index>> repeat_ops;
    shape_pos = 0;
    for (size_t i = 0; i < input_spec.size(); ++i) {
        const auto& item = input_spec[i];
        if (item.is_ellipsis) {
            for (int j = 0; j < ellipsis_dims; ++j) {
                intermediate_shape.push_back(in_shape[shape_pos]);
                intermediate_axes.push_back("batch_" + std::to_string(j));
                shape_pos++;
            }
        } else if (item.name == "1") {
            intermediate_shape.push_back(1);
            std::string out_axis = output_spec[i].name;
            Index new_size = axes_lengths.at(out_axis);
            intermediate_axes.push_back(out_axis);
            repeat_ops.emplace_back(static_cast<int>(intermediate_shape.size()-1), new_size);
            shape_pos++;
        } else if (item.is_group) {
            Index group_size = in_shape[shape_pos++];
            std::vector<std::string> group_axes;
            while (i < input_spec.size() && input_spec[i].is_group) {
                group_axes.push_back(input_spec[i].name);
                i++;
            }
            i--;
            Index known_product = 1;
            int unknown_count = 0;
            std::string unknown_axis;
            for (const auto& ax : group_axes) {
                if (axes_lengths.find(ax) != axes_lengths.end()) {
                    known_product *= axes_lengths[ax];
                } else {
                    unknown_count++;
                    unknown_axis = ax;
                }
            }
            if (unknown_count > 1)
                throw std::runtime_error("Too many unknown sizes in group");
            else if (unknown_count == 1)
                axis_sizes[unknown_axis] = group_size / known_product;
            else if (known_product != group_size)
                throw std::runtime_error("Group size mismatch");
            for (const auto& ax : group_axes) {
                intermediate_shape.push_back(axis_sizes[ax]);
                intermediate_axes.push_back(ax);
            }
        } else {
            intermediate_shape.push_back(axis_sizes[item.name]);
            intermediate_axes.push_back(item.name);
            shape_pos++;
        }
    }

    Eigen::DSizes<Index, 10> intermediate_dims;
    std::fill(intermediate_dims.begin(), intermediate_dims.end(), 1);
    int inter_rank = static_cast<int>(intermediate_shape.size());
    for (int i = 0; i < inter_rank; ++i)
        intermediate_dims[i] = intermediate_shape[i];

    auto reshaped = tensor_map.reshape(intermediate_dims);

    for (const auto& op : repeat_ops) {
        int axis = op.first;
        Index repeats = op.second;
        Eigen::DSizes<Index, 10> reps;
        std::fill(reps.begin(), reps.end(), 1);
        reps[axis] = repeats;
        reshaped = reshaped.broadcast(reps);
        intermediate_dims[axis] *= repeats; 
    }

    std::vector<Index> perm_vec;
    std::vector<Index> new_shape;
    for (size_t i = 0; i < output_spec.size(); ++i) {
        const auto& item = output_spec[i];
        if (item.is_ellipsis) {
            for (int j = 0; j < ellipsis_dims; ++j) {
                auto it = std::find(intermediate_axes.begin(), intermediate_axes.end(),
                                    "batch_" + std::to_string(j));
                Index idx = std::distance(intermediate_axes.begin(), it);
                perm_vec.push_back(idx);
                new_shape.push_back(intermediate_shape[idx]);
            }
        } else if (!item.is_group) {
            auto it = std::find(intermediate_axes.begin(), intermediate_axes.end(), item.name);
            Index idx = std::distance(intermediate_axes.begin(), it);
            perm_vec.push_back(idx);
            new_shape.push_back(intermediate_shape[idx]);
        } else {
            Index group_size = 1;
            std::vector<std::string> group_axes;
            while (i < output_spec.size() && output_spec[i].is_group) {
                group_axes.push_back(output_spec[i].name);
                auto it = std::find(intermediate_axes.begin(), intermediate_axes.end(), output_spec[i].name);
                perm_vec.push_back(std::distance(intermediate_axes.begin(), it));
                i++;
            }
            i--;
            for (const auto& ax : group_axes)
                group_size *= axis_sizes[ax];
            new_shape.push_back(group_size);
        }
    }

    Eigen::DSizes<Index, 10> final_dims;
    std::fill(final_dims.begin(), final_dims.end(), 1);
    int out_rank = static_cast<int>(new_shape.size());
    for (int i = 0; i < out_rank; ++i)
        final_dims[i] = new_shape[i];

    Eigen::array<Index, 10> perm_fixed;
    for (int i = 0; i < 10; ++i)
        perm_fixed[i] = i;
    for (int i = 0; i < static_cast<int>(perm_vec.size()); ++i)
        perm_fixed[i] = perm_vec[i];

    auto shuffled = reshaped.shuffle(perm_fixed);
    auto final_view = shuffled.reshape(final_dims);

    TensorFixed result(final_dims);
    result = final_view.eval();

    return result;
}

PYBIND11_MODULE(eigen_backend, m) {
    m.def("rearrange_eigen", &rearrange, "Rearrange tensor using Eigen",
          py::arg("tensor"), py::arg("input_spec"), py::arg("output_spec"), py::arg("axes_lengths"));
}
