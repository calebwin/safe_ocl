# safe_ocl

[![Gitter](https://badges.gitter.im/talk-about-emu/thoughts.svg)](https://gitter.im/talk-about-emu/thoughts?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)
[![](http://meritbadge.herokuapp.com/safe_ocl)](https://crates.io/crates/safe_ocl)
[![](https://docs.rs/safe_ocl/badge.svg)](https://docs.rs/safe_ocl)

This crate introduces zero-cost wrapper types for safe OpenCL. There are 2 wrapper types it introduces...

- `MapKernel`
- `MapProgram`

Currently, this is quite a limited set of types. Some of its limitations...

- Only supports map computation
- Only supports binary arithmetic operators
- Only safe for single-threaded
- Only safe for single-GPU
- Only safe under pre-condition that the read/write flag for buffer is correct

This is really just a base for adding new things like generics, new wrapper types, subtypes that can eliminate undefined behavior in OpenCL usage.
