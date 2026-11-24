#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

struct Evaluator {
    virtual ~Evaluator() = default;
    virtual double compute(int n, const double* x, const int* numbers, const double* cell, const bool* pbc,
                           double* forces, double* virial, double* energies) = 0;
    // MACE-Field: uniform field eps (the model's electric_field input, V/A); P = -dE/deps (e*A, whole cell)
    virtual bool has_field_coupling() const { return false; }
    virtual void set_electric_field(const double* e) {
        if (e[0] != 0 or e[1] != 0 or e[2] != 0)
            throw std::runtime_error("MACEFoundation: electric fields need a MACE-Field model on the CUDA device");
    }
    virtual void polarization(double* p) const { p[0] = p[1] = p[2] = 0; }
};

std::unique_ptr<Evaluator> mace_foundation_cpu(const std::string& file, const std::string& dtype, double skin);
std::unique_ptr<Evaluator> mace_foundation_cuda(const std::string& file, const std::string& dtype, double skin);

namespace mace_foundation {

struct NeighborList { std::vector<int> start, j, S; };
void neighbor_list(int n, const double* x, const double* cell, const bool* pbc, double cutoff, NeighborList& nl);

}  // namespace mace_foundation
