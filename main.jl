using QuantumOptics, CairoMakie


Npoints = 160
Npointsy = 120

xmin = -30
xmax = 50
b_position = PositionBasis(xmin, xmax, Npoints)
b_momentum = MomentumBasis(b_position)

ymin = -20
ymax = 20
b_positiony = PositionBasis(ymin, ymax, Npointsy)
b_momentumy = MomentumBasis(b_positiony)


b_comp_x = b_position ⊗ b_positiony
b_comp_p = b_momentum ⊗ b_momentumy

Txp = transform(b_comp_x, b_comp_p)
Tpx = transform(b_comp_p, b_comp_x)


px = momentum(b_momentum)
py = momentum(b_momentumy)


Hkinx = LazyTensor(b_comp_p, [1, 2], (px^2/2, one(b_momentumy)))
Hkiny = LazyTensor(b_comp_p, [1, 2], (one(b_momentum), py^2/2))

Hkinx_FFT = LazyProduct(Txp, Hkinx, Tpx)
Hkiny_FFT = LazyProduct(Txp, Hkiny, Tpx)


potential(x,y) = exp(-(x^2 + y^2)/30.0)
V = potentialoperator(b_comp_x, potential)


H = LazySum(Hkinx_FFT, Hkiny_FFT, V)


x0 = -10
y0 = -5
p0_x = 1.5
p0_y = 0.5
σ = 2.0

ψx = gaussianstate(b_position, x0, p0_x, σ)
ψy = gaussianstate(b_positiony, y0, p0_y, σ)
ψ = ψx ⊗ ψy

T = collect(0.0:0.1:20.0)
tout, ψt = timeevolution.schroedinger(T, ψ, H)


frame = Observable(1)

fig, ax, hm = heatmap(@lift(reshape(abs2.(ψt[$frame].data),(Npoints, Npointsy))))

record(fig, "hm3.mp4", 1:length(ψt); framerate = 30) do t
    frame[] = t
end
