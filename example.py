class TissueSlab:
    def __init__(self, name, D, slices, m_value_max, m_value_min):
        self.name = name
        self.slices = slices
        self.D = D
        self.slab = np.zeros(slices) # Start empty
        
        # Create a unique limit line for this tissue type
        self.m_values = np.linspace(m_value_max, m_value_min, slices)
        
        # Stability Constant (k)
        dx = 1.0
        self.k = (D * dt) / (dx**2)
        if self.k > 0.5: raise ValueError(f"Stability Error in {name}")

    def update(self, boundary_pressure):
        # 1. Blood touches the surface
        self.slab[0] = boundary_pressure
        
        # 2. Diffusion
        self.slab[1:-1] += self.k * (self.slab[:-2] - 2*self.slab[1:-1] + self.slab[2:])
        
        # 3. No-Flux at core
        self.slab[-1] = self.slab[-2]

    def get_risk(self):
        # Returns percentage of limit (1.0 = BENT)
        # We divide current pressure by the allowed M-Value
        risk_array = self.slab / self.m_values
        return np.max(risk_array) # The most dangerous slice defines the risk

# --- CONFIGURATION ---
# We model 3 distinct body parts
tissues = [
    # Fast, Sensitive (Spine)
    TissueSlab("Spine", D=0.002, slices=20, m_value_max=3.0, m_value_min=1.2),
    
    # Medium (Muscle)
    TissueSlab("Muscle", D=0.0005, slices=40, m_value_max=3.5, m_value_min=1.5),
    
    # Slow, Robust but traps gas (Joints)
    TissueSlab("Joints", D=0.0001, slices=60, m_value_max=4.0, m_value_min=2.0)
]

# --- SIMULATION LOOP ---
current_gas_pressure = 4.0 # e.g., 30m depth

for t in range(time_steps):
    overall_risk = 0
    limiting_tissue = ""

    for tissue in tissues:
        tissue.update(current_gas_pressure)
        
        # Check who is closest to exploding
        risk = tissue.get_risk()
        if risk > overall_risk:
            overall_risk = risk
            limiting_tissue = tissue.name

    if overall_risk >= 1.0:
        print(f"BENT! Failed in: {limiting_tissue}")
        break
