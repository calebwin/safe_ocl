# safe_ocl

[![Gitter](https://badges.gitter.im/talk-about-emu/thoughts.svg)](https://gitter.im/talk-about-emu/thoughts?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)
[![](http://meritbadge.herokuapp.com/safe_ocl)](https://crates.io/crates/safe_ocl)
[![](https://docs.rs/safe_ocl/badge.svg)](https://docs.rs/safe_ocl)

## about

This crate introduces zero-cost wrapper types for safe OpenCL. There are 2 wrapper types it introduces...

- `MapKernel`
- `MapProgram`

Currently, this is quite a limited set of types. Some of its limitations...

- Only supports map computation
- Only supports binary arithmetic operators
- Only safe for single-threaded
- Only safe for single-GPU
- Only safe under pre-condition that the read/write flag for buffer is correct

This is really just a skeleton for adding new things like generics, new wrapper types, subtypes that can eliminate undefined behavior in OpenCL usage.

## example

This is how you would normally implement an adding map operation.

```rust
let src = r#"
    __kernel void add(__global float* buffer, float scalar) {
        buffer[get_global_id(0)] += scalar;
    }
"#;

// (1) Define which platform and device(s) to use. Create a context,
// queue, and program then define some dims (compare to step 1 above).
let platform = Platform::default();
let device = Device::first(platform).unwrap();
let context = Context::builder()
    .platform(platform)
    .devices(device.clone())
    .build().unwrap();
let program = Program::builder()
    .devices(device)
    .src(src)
    .build(&context).unwrap();
let queue = Queue::new(&context, device, None).unwrap();
let dims = 1 << 20;
// [NOTE]: At this point we could manually assemble a ProQue by calling:
// `ProQue::new(context, queue, program, Some(dims))`. One might want to
// do this when only one program and queue are all that's needed. Wrapping
// it up into a single struct makes passing it around simpler.

// (2) Create a `Buffer`:
let buffer = Buffer::<f32>::builder()
    .queue(queue.clone())
    .flags(flags::MEM_READ_WRITE)
    .len(dims)
    .fill_val(0f32)
    .build().unwrap();

// (3) Create a kernel with arguments matching those in the source above:
let kernel = Kernel::builder()
    .program(&program)
    .name("add")
    .queue(queue.clone())
    .global_work_size(dims)
    .arg(&buffer)
    .arg(&10.0f32)
    .build().unwrap();

// (4) Run the kernel (default parameters shown for demonstration purposes):
unsafe {
    kernel.cmd()
        .queue(&queue)
        .global_work_offset(kernel.default_global_work_offset())
        .global_work_size(dims)
        .local_work_size(kernel.default_local_work_size())
        .enq().unwrap();
}

// (5) Read results from the device into a vector (`::block` not shown):
let mut vec = vec![0.0f32; dims];
buffer.cmd()
    .queue(&queue)
    .offset(0)
    .read(&mut vec)
    .enq().unwrap();

assert_eq!(vec, vec![10.0f32; dims]);
```

This is how you do it with the above types.

```rust
// (1) Define which platform and device(s) to use. Create a context,
// queue, and program then define some dims (compare to step 1 above).
let platform = Platform::default();
let device = Device::first(platform).unwrap();
let context = Context::builder()
    .platform(platform)
    .devices(device.clone())
    .build().unwrap();
let program = MapProgram::from(device, Op::Add, &context).unwrap();
let queue = Queue::new(&context, device, None).unwrap();
let dims = 1 << 20;
// [NOTE]: At this point we could manually assemble a ProQue by calling:
// `ProQue::new(context, queue, program, Some(dims))`. One might want to
// do this when only one program and queue are all that's needed. Wrapping
// it up into a single struct makes passing it around simpler.

// (2) Create a `Buffer`:
let buffer = Buffer::<f32>::builder()
    .queue(queue.clone())
    .flags(flags::MEM_READ_WRITE) // TODO ensure buffer is read-write
    .len(dims)
    .fill_val(0f32)
    .build().unwrap();

// (3) Create a kernel with arguments matching those in the source above:
let kernel = MapKernel::from(&program, queue.clone(), &buffer, &10.0f32).unwrap();

// (4) Run the kernel (default parameters shown for demonstration purposes):
kernel.cmd_enq(&queue);

// (5) Read results from the device into a vector (`::block` not shown):
let mut vec = vec![0.0f32; dims];
buffer.cmd()
    .queue(&queue)
    .offset(0)
    .read(&mut vec)
    .enq().unwrap();

assert_eq!(vec, vec![10.0f32; dims]);
```
