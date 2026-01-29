// Attention Bridge - Connects Electron to OpenCog Attention Allocation

#include <napi.h>
#include <iostream>
#include <vector>
#include <algorithm>

namespace attention_bridge {

// Stimulate atom with STI
Napi::Value StimulateAtom(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    
    if (info.Length() < 2 || !info[0].IsNumber() || !info[1].IsNumber()) {
        Napi::TypeError::New(env, "Expected atomId and amount as numbers")
            .ThrowAsJavaScriptException();
        return env.Null();
    }
    
    int atomId = info[0].As<Napi::Number>().Int32Value();
    int amount = info[1].As<Napi::Number>().Int32Value();
    
    std::cout << "Stimulating atom " << atomId << " with " << amount << " STI" << std::endl;
    
    // In real implementation, update atom's STI in AtomSpace
    
    return Napi::Boolean::New(env, true);
}

// Get attentional focus
Napi::Value GetAttentionalFocus(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    
    std::cout << "Getting attentional focus" << std::endl;
    
    // Mock focus (in real implementation, query AtomSpace for high-STI atoms)
    Napi::Array focus = Napi::Array::New(env);
    
    // Create mock focus atoms
    for (int i = 0; i < 5; i++) {
        Napi::Object atom = Napi::Object::New(env);
        atom.Set("id", Napi::Number::New(env, i + 1));
        atom.Set("name", Napi::String::New(env, "atom_" + std::to_string(i + 1)));
        atom.Set("sti", Napi::Number::New(env, 100 - (i * 10)));
        focus.Set(i, atom);
    }
    
    return focus;
}

} // namespace attention_bridge
