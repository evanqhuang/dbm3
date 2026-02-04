import numpy as np
import matplotlib.pyplot as plt

# --- USER CONFIGURATION ---
surface_altitude_m = 0  # Altitude of the water surface (0m = Sea Level)
bottom_depth_m = 30  # Depth of the dive in meters
dive_time_min = 30  # Bottom time in minutes
fO2 = 0.21  # Breathing gas O2 fraction (Air = 0.21)

# --- M-VALUE PARAMETERS (Slab-based) ---
# For a slab model, we create an array of M-values for each slice.
# Surface/Fast slices (0) have HIGH tolerance (gas leaves quickly)
# Core/Slow slices (49) have LOW tolerance (gas trapped longer)
M_SURFACE_SLICE = 3.0  # Max tolerable ppN2 at slice 0 (fast tissue)
M_CORE_SLICE = 1.5  # Max tolerable ppN2 at slice 49 (slow tissue)

# --- PHYSICS CONSTANTS ---
slices = 50  # Resolution (layers in the slab)
D = 0.1  # Diffusion coefficient (Speed of gas moving)
dt = 0.5  # Time step in seconds (smaller = more precise)
dx = 1.0  # Distance between slices (arbitrary units)

# --- PERMEABILITY BARRIER (The "Hidden" Knob) ---
# Controls delay at blood-tissue interface. Higher = faster equilibration.
# Set to None for perfect perfusion (instant, original behavior)
# Typical range: 0.001 (slow barrier) to 0.1 (fast barrier)
permeability = 0.0003  # Flux = Permeability * (Blood_Pressure - Surface_Pressure)

# Stability Check: The finite difference method explodes if k > 0.5
k = (D * dt) / (dx**2)
if k > 0.5:
    raise ValueError(
        f"Stability warning! k={k:.4f} is > 0.5. Reduce dt or increase dx."
    )

# Create M-Value Array (the "Death Line")
# Smooth curve from high tolerance (surface) to low tolerance (core)
m_values = np.linspace(M_SURFACE_SLICE, M_CORE_SLICE, slices)


# --- PRESSURE CALCULATIONS ---
def get_atmospheric_pressure(altitude_m):
    # Standard Barometric Formula to estimate pressure at altitude
    # P = P0 * (1 - L*h/T0)^(gM/RL)
    if altitude_m < 0:
        return 1.01325  # Clamp to sea level if negative
    return 1.01325 * (1 - 2.25577e-5 * altitude_m) ** 5.25588


def get_hydrostatic_pressure(depth_m):
    # Standard rule: 1 bar per 10m of depth
    return depth_m / 10.0


def get_ndl(current_slab, current_depth, dt, D, m_values, permeability=None):
    """
    Simulates the future to find how many minutes until we hit the limit.
    Uses a "shadow simulation" - clones state and runs forward in time.
    """
    # 1. Create a "Shadow Slab" so we don't mess up the real dive
    shadow_slab = current_slab.copy()

    # Calculate pressure at current depth (we are staying here)
    p_bottom = get_atmospheric_pressure(0) + (current_depth / 10.0)
    ppN2_bottom = p_bottom * (1 - fO2)

    # Stability constant
    k_shadow = (D * dt) / (dx**2)

    # 2. Look ahead up to 99 minutes
    for minute in range(100):
        # Run simulation for 60 seconds (one minute block)
        steps_per_min = int(60 / dt)

        for _ in range(steps_per_min):
            # Update Boundary
            if permeability is None:
                shadow_slab[0] = ppN2_bottom
            else:
                flux = permeability * (ppN2_bottom - shadow_slab[0])
                shadow_slab[0] += flux * dt
            # Diffuse
            shadow_slab[1:-1] += k_shadow * (
                shadow_slab[:-2] - 2 * shadow_slab[1:-1] + shadow_slab[2:]
            )
            # No-flux deep end
            shadow_slab[-1] = shadow_slab[-2]

        # 3. The NDL Check:
        # If ANY slice in the shadow slab is higher than its matching M-Value
        if np.any(shadow_slab > m_values):
            return minute  # We found the limit!

    return 99  # More than 99 mins (essentially unlimited)


# 1. Calculate Surface Pressure (Start)
p_surface_bar = get_atmospheric_pressure(surface_altitude_m)
ppN2_surface = p_surface_bar * (1 - fO2)

# 2. Calculate Bottom Pressure (Dive)
p_total_bottom = p_surface_bar + get_hydrostatic_pressure(bottom_depth_m)
ppN2_bottom = p_total_bottom * (1 - fO2)

print("--- DIVE PLAN ---")
print(f"Altitude: {surface_altitude_m}m (Atm Pressure: {p_surface_bar:.2f} bar)")
print(f"Depth: {bottom_depth_m}m (Total Pressure: {p_total_bottom:.2f} bar)")
print(f"Breathing ppN2: Surface={ppN2_surface:.2f}, Bottom={ppN2_bottom:.2f}")

# --- INITIALIZATION ---
# The tissue starts fully saturated at surface pressure
slab = np.full(slices, ppN2_surface)
history = [slab.copy()]  # Save T=0 state
max_tissue_load = []  # Track max pressure across all slices
time_points_min = []  # Time in minutes for plotting

# --- SIMULATION LOOP ---
# Convert minutes to seconds
dive_steps = int((dive_time_min * 60) / dt)

for t in range(dive_steps):
    # 1. Update Boundary Condition (Blood-Tissue Interface)
    if permeability is None:
        # Perfect Perfusion: surface instantly matches blood
        slab[0] = ppN2_bottom
    else:
        # Permeability Barrier: adds delay at interface
        # Flux = Permeability * (Blood_Pressure - Surface_Pressure)
        flux = permeability * (ppN2_bottom - slab[0])
        slab[0] += flux * dt

    # 2. Diffusion (Finite Difference Method)
    # The math: New = Old + k * (Left_Neighbor - 2*Me + Right_Neighbor)
    # We use vectorization (slab[1:-1]) to do all slices at once for speed
    slab[1:-1] += k * (slab[:-2] - 2 * slab[1:-1] + slab[2:])

    # 3. No-Flux Boundary at the deep end
    # (Gas hits the back of the cartilage and stops, or bounces back)
    slab[-1] = slab[-2]

    # Save state every 60 seconds (simulated time)
    if (t * dt) % 60 < dt:
        history.append(slab.copy())
        max_tissue_load.append(np.max(slab))
        time_points_min.append((t * dt) / 60)

# --- M-VALUE CHECK (Slab-based) ---
# In the slab model, each slice has its own M-value limit.
# NDL is exceeded when ANY slice exceeds its corresponding M-value.

print("\n--- M-VALUE LIMITS (Slab Array) ---")
print(f"M-value at Slice 0 (fast/surface): {m_values[0]:.2f} bar")
print(f"M-value at Slice {slices - 1} (slow/core): {m_values[-1]:.2f} bar")

# Check which slice is closest to its limit
margin = m_values - slab  # How much headroom each slice has
min_margin_idx = np.argmin(margin)
min_margin = margin[min_margin_idx]

print(f"\nCritical slice: {min_margin_idx} (margin: {min_margin:.3f} bar)")

# Check if any slice exceeded its M-value
exceeded = np.any(slab > m_values)
if exceeded:
    exceeded_slices = np.where(slab > m_values)[0]
    print(f"\n⚠️  WARNING: Slices {exceeded_slices} EXCEED their M-values!")
    print("   Mandatory decompression stops required before surfacing.")
else:
    # Calculate NDL from current state
    current_ndl = get_ndl(slab, bottom_depth_m, dt, D, m_values, permeability)
    print(f"\n✓ All slices within limits. NDL remaining: {current_ndl} min")

# --- VISUALIZATION ---
history_array = np.array(history)
max_tissue_load = np.array(max_tissue_load)
time_points_min = np.array(time_points_min)

fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(10, 10), gridspec_kw={"height_ratios": [2, 1]}
)

# --- TOP PLOT: Heatmap ---
im = ax1.imshow(
    history_array.T,
    aspect="auto",
    cmap="hot",
    origin="lower",
    extent=[0, dive_time_min, 0, slices],
)
plt.colorbar(im, ax=ax1, label="Partial Pressure N2 (bar)")
ax1.set_xlabel("Dive Time (minutes)")
ax1.set_ylabel("Depth into Tissue Slab (arbitrary units)")
ax1.set_title(f"Nitrogen Diffusion: {bottom_depth_m}m Dive for {dive_time_min} mins")

# Add contour lines for readability
ax1.contour(
    history_array.T,
    levels=10,
    colors="black",
    alpha=0.3,
    origin="lower",
    extent=[0, dive_time_min, 0, slices],
)
# --- BOTTOM PLOT: Slab Profile vs M-Value Array ---
# Plot the final slab state against the M-value limit curve
slice_indices = np.arange(slices)
ax2.plot(slice_indices, slab, "b-", linewidth=2, label="Current Gas Load")
ax2.plot(slice_indices, m_values, "r--", linewidth=2, label="M-Value Limit")
ax2.axhline(
    y=ppN2_surface,
    color="green",
    linestyle="-",
    linewidth=1,
    alpha=0.5,
    label=f"Surface ppN2 = {ppN2_surface:.2f} bar",
)
ax2.axhline(
    y=ppN2_bottom,
    color="orange",
    linestyle=":",
    linewidth=1,
    alpha=0.5,
    label=f"Bottom ppN2 = {ppN2_bottom:.2f} bar",
)

# Shade the danger zone (above M-value curve)
ax2.fill_between(
    slice_indices,
    m_values,
    np.max(m_values) * 1.2,
    color="red",
    alpha=0.1,
    label="Deco Required Zone",
)

# Mark any slices that exceeded their M-value
exceeded_slices = np.where(slab > m_values)[0]
if len(exceeded_slices) > 0:
    ax2.scatter(
        exceeded_slices,
        slab[exceeded_slices],
        color="red",
        s=50,
        zorder=5,
        label="Exceeded!",
    )
    ax2.annotate(
        f"Limit exceeded\nat slice {exceeded_slices[0]}",
        xy=(exceeded_slices[0], slab[exceeded_slices[0]]),
        xytext=(exceeded_slices[0] + 5, slab[exceeded_slices[0]] + 0.2),
        fontsize=9,
        color="red",
        arrowprops=dict(arrowstyle="->", color="red"),
    )

ax2.set_xlabel("Slice Index (0=Surface/Fast, 49=Core/Slow)")
ax2.set_ylabel("Partial Pressure N2 (bar)")
ax2.set_title("Slab Gas Load vs. M-Value Array (after dive)")
ax2.legend(loc="upper right")
ax2.set_xlim(0, slices - 1)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
