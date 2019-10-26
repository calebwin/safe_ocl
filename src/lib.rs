use ocl::{Queue, Buffer, Kernel, Context, Program, builders::DeviceSpecifier, error::Result};

#[derive(PartialEq, Eq)]
/// An operation that can be used to map over data
pub enum Op {
	Add,
	Min,
	Mul,
	Div,
	Mod,
	None
}

/// A safe wrapper type for Program
pub struct MapProgram(Program);
/// A safe wrapper type for Kernel
pub struct MapKernel(Kernel, usize);

impl MapProgram {
	/// Creates a new program the uses given device to apply given mapping operation over data with given context
	pub fn from<D: Into<DeviceSpecifier>>(devices: D, op: Op, context: &Context) -> Result<Self> {
		let src = if op == Op::None {
			String::from("__kernel void __main__(__global float* buffer, float scalar) {}")
		} else {
			format!(r#"
				__kernel void __main__(__global float* buffer, float scalar) {{
					buffer[get_global_id(0)] {}= scalar;
				}}
			"#, match op {
				Op::Add => "+",
				Op::Min => "-",
				Op::Mul => "*",
				Op::Div => "/",
				Op::Mod => "%",
				Op::None => panic!("creating program failed")
			})
		};

		Program::builder()
			.devices(devices)
			.src(src)
			.build(&context)
			.map(|program| Self(program))
	}
}

impl MapKernel {
	/// Creates a new kernel the runs given program using given queue to map an operation over given buffer with given value
	fn from(program: &MapProgram, queue: Queue, buffer: &Buffer<f32>, val: &f32) -> Result<Self> {
		let buffer_len = buffer.len();
		Kernel::builder()
		    .program(&program.0)
		    .name("__main__")
		    .queue(queue.clone())
		    .global_work_size(buffer_len)
		    .arg(buffer)
		    .arg(val)
		    .build()
		    .map(|kernel| Self(kernel, buffer_len))
	}

	/// Executes the kernel
	fn cmd_enq(&self, queue: &Queue) {
		unsafe {
		    self.0.cmd()
		        .queue(&queue)
		        .global_work_offset(self.0.default_global_work_offset())
		        .global_work_size(self.1)
		        .local_work_size(self.0.default_local_work_size())
		        .enq().unwrap();
		}
	}
}

#[cfg(test)]
mod tests {
    use super::*;
	use ocl::{flags, Platform, Device, Context, Queue, Program, Buffer, Kernel};

    #[test]
    fn test_add_unsafe() {
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
    }

    #[test]
    fn test_add() {
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
    }
}
