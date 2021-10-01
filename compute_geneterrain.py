"""
Example compute shader that does ... nothing but copy a value from one
buffer into another.
"""
RUST_BACKTRACE = 1

from array import array
import wgpu
import wgpu.backends.rs  # Select backend
from wgpu.utils import compute_with_buffers  # Convenience function
import numpy as np
from matplotlib import cm
from PIL import Image
import time

N = 2000

# %% Shader and data

terrain_source = """
// compute terrain wgsl
struct Node {
    value : f32;
    x : f32;
    y : f32;
    size : f32;
};
[[block]] struct Nodes {
    nodes : array<Node>;
};
[[block]] struct Uniforms {
  image_width : f32;
  image_height : f32;
  nodes_length : f32;
  width_factor : f32;
  view_box : vec4<f32>;
};
[[block]] struct Pixels {
    pixels : array<f32>;
};

[[group(0), binding(0)]] var<storage, read> nodes : Nodes;
[[group(0), binding(1)]] var<uniform> uniforms : Uniforms;
[[group(0), binding(2)]] var<storage, read_write> pixels : Pixels;

[[stage(compute), workgroup_size(1, 1, 1)]]
fn main([[builtin(global_invocation_id)]] global_id : vec3<u32>) {
    var pixel_index : u32 = global_id.x + global_id.y * u32(uniforms.image_width);
    var x : f32 = f32(global_id.x) / uniforms.image_width;
    var y : f32 = f32(global_id.y) / uniforms.image_height;
    x = x * (uniforms.view_box.z - uniforms.view_box.x) + uniforms.view_box.x;
    y = y * (uniforms.view_box.w - uniforms.view_box.y) + uniforms.view_box.y;
    var value : f32 = 0.0;

    for (var i : u32 = 0u; i < u32(uniforms.nodes_length); i = i + 1u) {
        var sqrDistance : f32 = (x - nodes.nodes[i].x) * (x - nodes.nodes[i].x) + (y - nodes.nodes[i].y) * (y - nodes.nodes[i].y);
        value = value + nodes.nodes[i].value / (sqrDistance * uniforms.width_factor + 1.0);
    }
    pixels.pixels[pixel_index] = value;
}
"""
# Create input data as a memoryview
n = 20
data = memoryview(bytearray(n * 4)).cast("i")
for i in range(n):
    data[i] = i
test_data = memoryview(bytearray(8 * 4)).cast("f")
test_data[0] = N
test_data[1] = N

# %% The short version, using memoryview

# The first arg is the input data, per binding
# The second arg are the ouput types, per binding
# out = compute_with_buffers({0: data}, {1: (n, "i")}, shader_source, n=n)

# The result is a dict matching the output types
# Select data from buffer at binding 1
# result = out[1]
# print(result.tolist())


# %% The short version, using numpy

# import numpy as np
#
# numpy_data = np.frombuffer(data, np.int32)
# out = compute_with_buffers({0: numpy_data}, {1: numpy_data.nbytes}, compute_shader, n=n)
# result = np.frombuffer(out[1], dtype=np.int32)
# print(result)


# %% The long version using the wgpu API

# Create device and shader object
device = wgpu.utils.get_default_device()
cshader = device.create_shader_module(code=terrain_source)

# Load gene terrain files
f = open("SingleTerrainExample/Expression.txt")
nodeID_to_value = {part.split("\t")[0]: float(part.split("\t")[1]) for part in f.read().split("\n")}
f = open("SingleTerrainExample/Layout.txt")
nodes = []
for node in f.read().split("\n"):
    parts = node.split("\t")
    if nodeID_to_value.get(parts[0]):
        nodes.extend([nodeID_to_value[parts[0]], float(parts[1]), float(parts[2]), float(parts[3])])
test_data[2] = len(nodes) / 4
test_data[3] = N
test_data[4] = 0
test_data[5] = 0
test_data[6] = 1
test_data[7] = 1
nodes = memoryview(array("f", nodes))

# Create buffer objects, input buffer is mapped.
buffer1 = device.create_buffer_with_data(data=nodes, usage=wgpu.BufferUsage.STORAGE)
buffer2 = device.create_buffer(
    size= N * N * 4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
)
uniforms = device.create_buffer_with_data(
    data=test_data, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
)

# Setup layout and bindings
binding_layouts = [
    {
        "binding": 0,
        "visibility": wgpu.ShaderStage.COMPUTE,
        "buffer": {
            "type": wgpu.BufferBindingType.read_only_storage,
        },
    },
    {
        "binding": 1,
        "visibility":  wgpu.ShaderStage.COMPUTE,
        "buffer": {
            "type": wgpu.BufferBindingType.uniform,
        }
    },
    {
        "binding": 2,
        "visibility": wgpu.ShaderStage.COMPUTE,
        "buffer": {
            "type": wgpu.BufferBindingType.storage,
        },
    },
]
bindings = [
    {
        "binding": 0,
        "resource": {"buffer": buffer1, "offset": 0, "size": buffer1.size},
    },
    {
        "binding": 1,
        "resource": {"buffer": uniforms, "offset": 0, "size": uniforms.size}
    },
    {
        "binding": 2,
        "resource": {"buffer": buffer2, "offset": 0, "size": buffer2.size},
    },

]

# Put everything together
bind_group_layout = device.create_bind_group_layout(entries=binding_layouts)
pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[bind_group_layout])
bind_group = device.create_bind_group(layout=bind_group_layout, entries=bindings)


# Create and run the pipeline
compute_pipeline = device.create_compute_pipeline(
    layout=pipeline_layout,
    compute={"module": cshader, "entry_point": "main"},
)
start = time.time()
command_encoder = device.create_command_encoder()
compute_pass = command_encoder.begin_compute_pass()
compute_pass.set_pipeline(compute_pipeline)
compute_pass.set_bind_group(0, bind_group, [], 0, 999999)  # last 2 elements not used
compute_pass.dispatch(N, N, 1)  # x y z
compute_pass.end_pass()
device.queue.submit([command_encoder.finish()])
# Read result
# result = buffer2.read_data().cast("i")
result = device.queue.read_buffer(buffer2).cast("f")
end = time.time()
print(end-start)
result = result.tolist()
rmin = min(result)
rmax = max(result)
for i in range(len(result)):
    result[i] = (result[i] - rmin) / (rmax - rmin)
result = np.array(result).reshape((N, N))
result = np.flipud(result)
im = Image.fromarray(np.uint8(cm.rainbow(result) * 255))
im.show()