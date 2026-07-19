#pragma once

#include <vector>
#include <functional>
#include <string>
#include <set>
#include <iostream>
#include <cmath>

class Value {
public:
    float data;                           // actual number
    float grad;                           // gradient — shuru mein 0
    std::string op;                       // kaunse op se bana — debug ke liye
    std::vector<Value*> parents;          // kahan se aaya
    std::function<void()> backward_fn;   // chain rule — har op ke liye alag

    // constructor
    Value(float data, std::string op = "");

    // ─ Operations -> ye sab graph build karte hain 
    Value* add(Value* other);
    Value* mul(Value* other);
    Value* tanh_op();
    Value* pow_op(float exp);
    Value* neg();        // negate — subtract ke liye
    Value* sub(Value* other);
    Value* exp_op();     // e^x — loss ke liye baad mein
    Value* div_op(Value* other);   // a / b
    Value* log_op();               // ln(x)
    Value* relu();                 // max(0, x)
    Value* sigmoid();              // 1 / (1 + e^-x)

    // ── Backward pass ──
    void backward();     // poore graph pe gradients calculate karo

    // ── Utility ──
    void print() const;

private:
    // topological sort — DFS se
    void build_topo(Value* v,
                    std::set<Value*>& visited,
                    std::vector<Value*>& topo);
};