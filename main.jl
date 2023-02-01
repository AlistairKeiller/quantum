using QuantumOptics, GLMakie


Npoints = 160
Npointsy = 120

xmin = -20
xmax = 20
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

function potential(x,y)
    if -.1 < x < .1 && !(2 < y < 5 || -5 < y < -2)
        100
    else
        0
    end
end
V = potentialoperator(b_comp_x, potential)


H = LazySum(Hkinx_FFT, Hkiny_FFT, V)

f = Figure()

x0 = -5
y0 = 0
p0_x = 3
p0_y = 0
σ = 2

sg = SliderGrid(f[2,1],
    (label = "x0", range = -10:0.1:0, startvalue = x0),
    (label = "y0", range = -5:0.1:5, startvalue = y0),
    (label = "p0_x", range = 0:0.1:5, startvalue = p0_x),
    (label = "p0_y", range = -2:0.1:2, startvalue = p0_y),
    (label = "σ", range = 0:0.1:5, startvalue = σ),
)
simulate = Button(f[3,1]; label = "simulate", tellwidth = false)

ψx = gaussianstate(b_position, x0, p0_x, σ)
ψy = gaussianstate(b_positiony, y0, p0_y, σ)
ψ = ψx ⊗ ψy

T = collect(0.0:0.1:5.0)
tout, ψt = timeevolution.schroedinger(T, ψ, H)

frame = Observable(1)

heatmap(f[1,1][1,1],@lift(reshape(abs2.(ψt[$frame].data),(Npoints, Npointsy))))

on(simulate.clicks) do clicks
    ψx = gaussianstate(b_position, to_value(sg.sliders[1].value), to_value(sg.sliders[3].value), to_value(sg.sliders[5].value))
    ψy = gaussianstate(b_positiony, to_value(sg.sliders[2].value), to_value(sg.sliders[4].value), to_value(sg.sliders[5].value))
    ψ = ψx ⊗ ψy

    tout, ψt = timeevolution.schroedinger(T, ψ, H)

    @async for i in 1:length(ψt)
        frame[] = i
        sleep(1/30)
    end
end
