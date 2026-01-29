// AtomSpace Bridge - Connects Electron to OpenCog AtomSpace

#include <napi.h>
#include <iostream>
#include <string>
#include <vector>
#include <map>

namespace atomspace_bridge {

// Simplified atom structure (in real implementation, use OpenCog types)
struct Atom {
    int id;
    std::string type;
    std::string name;
    double strength;
    double confidence;
    int sti;
    int lti;
};

// Mock AtomSpace (replace with real OpenCog AtomSpace)
static std::map<int, Atom> g_atoms;
static int g_next_id = 1;

// Create atom in AtomSpace
Napi::Value CreateAtom(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    
    // Validate arguments
    if (info.Length() < 4) {
        Napi::TypeError::New(env, "Expected 4 arguments: type, name, strength, confidence")
            .ThrowAsJavaScriptException();
        return env.Null();
    }
    
    if (!info[0].IsString() || !info[1].IsString() || 
        !info[2].IsNumber() || !info[3].IsNumber()) {
        Napi::TypeError::New(env, "Invalid argument types")
            .ThrowAsJavaScriptException();
        return env.Null();
    }
    
    // Extract arguments
    std::string type = info[0].As<Napi::String>().Utf8Value();
    std::string name = info[1].As<Napi::String>().Utf8Value();
    double strength = info[2].As<Napi::Number>().DoubleValue();
    double confidence = info[3].As<Napi::Number>().DoubleValue();
    
    // Create atom
    Atom atom;
    atom.id = g_next_id++;
    atom.type = type;
    atom.name = name;
    atom.strength = strength;
    atom.confidence = confidence;
    atom.sti = 0;
    atom.lti = 0;
    
    // Store in mock atomspace
    g_atoms[atom.id] = atom;
    
    std::cout << "Created atom: (" << type << " \"" << name << "\") [" << atom.id << "]" << std::endl;
    
    // Return atom ID
    return Napi::Number::New(env, atom.id);
}

// Get atom by ID
Napi::Value GetAtom(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    
    if (info.Length() < 1 || !info[0].IsNumber()) {
        Napi::TypeError::New(env, "Expected atom ID as number")
            .ThrowAsJavaScriptException();
        return env.Null();
    }
    
    int atomId = info[0].As<Napi::Number>().Int32Value();
    
    // Find atom
    auto it = g_atoms.find(atomId);
    if (it == g_atoms.end()) {
        Napi::Error::New(env, "Atom not found").ThrowAsJavaScriptException();
        return env.Null();
    }
    
    const Atom& atom = it->second;
    
    // Create JavaScript object
    Napi::Object result = Napi::Object::New(env);
    result.Set("id", Napi::Number::New(env, atom.id));
    result.Set("type", Napi::String::New(env, atom.type));
    result.Set("name", Napi::String::New(env, atom.name));
    
    Napi::Object tv = Napi::Object::New(env);
    tv.Set("strength", Napi::Number::New(env, atom.strength));
    tv.Set("confidence", Napi::Number::New(env, atom.confidence));
    result.Set("tv", tv);
    
    Napi::Object av = Napi::Object::New(env);
    av.Set("sti", Napi::Number::New(env, atom.sti));
    av.Set("lti", Napi::Number::New(env, atom.lti));
    av.Set("vlti", Napi::Number::New(env, 0));
    result.Set("av", av);
    
    return result;
}

// Query AtomSpace with pattern
Napi::Value QueryAtomSpace(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    
    if (info.Length() < 1 || !info[0].IsString()) {
        Napi::TypeError::New(env, "Expected pattern as string")
            .ThrowAsJavaScriptException();
        return env.Null();
    }
    
    std::string pattern = info[0].As<Napi::String>().Utf8Value();
    
    std::cout << "Querying AtomSpace with pattern: " << pattern << std::endl;
    
    // Mock query results (in real implementation, use pattern matcher)
    Napi::Array results = Napi::Array::New(env);
    
    int index = 0;
    for (const auto& pair : g_atoms) {
        const Atom& atom = pair.second;
        
        Napi::Object binding = Napi::Object::New(env);
        binding.Set("var", Napi::String::New(env, "$X"));
        
        Napi::Object atomObj = Napi::Object::New(env);
        atomObj.Set("id", Napi::Number::New(env, atom.id));
        atomObj.Set("type", Napi::String::New(env, atom.type));
        atomObj.Set("name", Napi::String::New(env, atom.name));
        
        binding.Set("atom", atomObj);
        results.Set(index++, binding);
        
        // Limit results for demo
        if (index >= 10) break;
    }
    
    return results;
}

// Get AtomSpace statistics
Napi::Value GetAtomSpaceStats(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    
    // Count atoms by type
    std::map<std::string, int> typeCounts;
    int nodeCount = 0;
    int linkCount = 0;
    
    for (const auto& pair : g_atoms) {
        const Atom& atom = pair.second;
        typeCounts[atom.type]++;
        
        if (atom.type.find("Node") != std::string::npos) {
            nodeCount++;
        } else if (atom.type.find("Link") != std::string::npos) {
            linkCount++;
        }
    }
    
    // Create result object
    Napi::Object stats = Napi::Object::New(env);
    stats.Set("totalAtoms", Napi::Number::New(env, g_atoms.size()));
    stats.Set("nodes", Napi::Number::New(env, nodeCount));
    stats.Set("links", Napi::Number::New(env, linkCount));
    
    // Type breakdown
    Napi::Object types = Napi::Object::New(env);
    for (const auto& pair : typeCounts) {
        types.Set(pair.first, Napi::Number::New(env, pair.second));
    }
    stats.Set("types", types);
    
    return stats;
}

} // namespace atomspace_bridge
