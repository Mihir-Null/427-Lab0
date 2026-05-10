# CMSC427 Lab 00 Report: Parametric Curves

This lab was implemented both in P5.js as well as in Rust with wgpu and winit. It was initially done in P5.js and then redone (albeit more simply) in wbgpu and rust as I learned that I found developing in webgl and javascript extremely painful. It ports over the parametric curve exercise by outlining a complete pipeline by generating curve vertices on the CPU, uploading them to a GPU vertex buffer, and drawing them with a line-strip pipeline, though only the make_rose_curve function and shader definition are relevant to the lab exercises. The rest of the code is the pipeline infrastructure necessary for wgpu and is largely boilerplate but has been annotated as well. 

I extensively used this [tutorial](https://sotrh.github.io/learn-wgpu/) and code contained therein to learn wgpu and implement all the labs. AI tools were used for understanding and mapping  but were not used for coding. VSCode extensions were used for linting and formatting.

## Parametric Equation

The curve is generated in `make_rose_curve`. For each sample point, the code evaluates a rose curve:

```rust
let r = (petals as f32 * angle).cos();
let x = r * angle.cos() * 0.8;
let y = r * angle.sin() * 0.8;
```

The project uses `5` petals and `2000` sample points. The high sample count smoothens out the curve as the line strip is rendered as many short segments. The final scale factor keeps the curve inside normalized device coordinates, so no camera or projection matrix is needed. Shading code simply ingests the rust code to output to WGSL

## Animation

The curve changes over time by adding a phase offset to the angle:

```rust
let phase = time * 0.8;
let angle = t * 2.0 * PI + phase;
```

Each frame computes elapsed time using `Instant::now()`, then regenerates the curve vertices, and finally writes them into the existing GPU buffer with `queue.write_buffer`. This ensures GPU resources aren't being reinitialized for every frame while still making the visible curve animate.

## Vertex Attributes and Color

Each vertex stores a two-dimensional position and an RGB color. The color is also generated procedurally from the angle:

```rust
let color = [
    (angle * 0.5).sin() * 0.5 + 0.5,
    0.2,
    (angle * 0.5).cos() * 0.5 + 0.5,
];
```

The sine and cosine terms are remapped into the `0` to `1` range (required by wgpu), producing a smooth color gradient around the curve. Changing these parameters can more clearly show interpolation on a per vertex basis.

## Render Pipeline

The render pipeline uses `wgpu::PrimitiveTopology::LineStrip`. This is the correct topology for a connected parametric curve because each vertex after the first extends the line from the previous vertex.

The WGSL shader is intentionally small. The vertex shader sends the 2D position directly to clip space, and the fragment shader returns the interpolated color. This keeps the walkthrough focused on procedural vertex generation and buffer updates.

## wgpu Porting Notes

The WebGL version of this exercise would usually update a JavaScript array or typed array and redraw the line. In this Rust/wgpu port, the same concept becomes a typed `Vertex` struct (not provided by default but following the pattern in the tutorial I was using), a `VERTEX | COPY_DST` GPU buffer, and a per-frame `queue.write_buffer` call.

## Result

The final result is an animated five-petal rose curve rendered as a colored line strip. It demonstrates how a mathematical curve can be sampled into vertices, updated over time, uploaded to the GPU, and displayed through a (relatively) minimal wgpu render pipeline that is used in future labs.
