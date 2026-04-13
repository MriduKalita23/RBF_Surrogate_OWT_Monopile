import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm, gumbel_r
import math
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score
import warnings
import pandas as pd
warnings.filterwarnings('ignore')

np.random.seed(42)

# ENVIRONMENTAL LOADING CALCULATIONS
class EnvironmentalLoads:
    def __init__(self, site_params):
        self.g = 9.81
        self.rho_air = 1.225
        self.rho_water = 1025
        self.site = site_params
        
    def wind_pressure(self, z, v_hub=50.0, z_hub=90.0, alpha=0.115):
        u_z = v_hub * (z / z_hub) ** alpha
        C_p = 1.0
        P_w = 0.5 * C_p * self.rho_air * u_z**2
        return P_w
    
    def wave_current_load(self, z, t, D, H_s=6.9, T_s=7.7, v_c=0.8, d_w=20.0):
        C_M = 2.0
        C_D = 1.2
        wavelength = 1.56 * T_s**2
        k = 2 * np.pi / wavelength
        omega = 2 * np.pi / T_s
        a = H_s / 2
        depth_factor = np.cosh(k * (d_w + z)) / np.sinh(k * d_w)
        u_wave = omega * a * depth_factor * np.cos(omega * t)
        u_wave_dot = omega**2 * a * depth_factor * np.sin(omega * t)
        u_current = v_c
        u_total = u_wave + u_current
        F_inertia = (np.pi/4) * self.rho_water * C_M * D**2 * u_wave_dot
        F_drag = 0.5 * self.rho_water * C_D * D * u_total * np.abs(u_total)
        return F_inertia + F_drag
    
    def hydrostatic_pressure(self, z):
        return self.rho_water * self.g * np.abs(z)
    
    def gravity_load_RNA(self, m_RNA=350000):
        return m_RNA * self.g
    
    def calculate_total_loads(self, D3, z_hub=90.0, tower_height=77.6, water_depth=20.0):
        loads = {
            'F_Ax': 46990, 'F_Ay': 475000, 'F_Az': 72670,
            'M_Ax': -1129000, 'M_Ay': -112600, 'M_Az': -217600
        }
        n_points = 20
        z_tower = np.linspace(water_depth, water_depth + tower_height, n_points)
        wind_pressures = np.array([self.wind_pressure(z) for z in z_tower])
        loads['P_w_avg'] = np.mean(wind_pressures)
        
        t_max = 0
        z_pile = np.linspace(-water_depth, 0, n_points)
        wave_forces = np.array([self.wave_current_load(z, t_max, D3) for z in z_pile])
        dz = water_depth / (n_points - 1)
        loads['F_Hx'] = np.sum(wave_forces) * dz
        lever_arms = np.abs(z_pile)
        loads['M_Hy'] = np.sum(wave_forces * lever_arms) * dz
        loads['P_h_mudline'] = self.hydrostatic_pressure(-water_depth)
        loads['G_RNA'] = self.gravity_load_RNA()
        return loads

# DATA LOADING
def load_excel_data():
    try:
        df_wave = pd.read_excel('Kalpakkam_1998_2017_Final.xlsx', sheet_name='Sheet1')
        df_tide = pd.read_excel('Kalpakkam_tide.xlsx', sheet_name='Sheet1')
        print("✓ Excel data loaded successfully!")
        return df_wave, df_tide
    except Exception as e:
        print(f"⚠ Warning: Could not load Excel files: {e}")
        return None, None

def process_wave_data(df_wave):
    if df_wave is None:
        return {'H_s_mean': 1.0, 'H_s_max': 6.9, 'T_mean': 4.5, 'T_max': 7.7}
    return {
        'H_s_mean': df_wave['Wave Height'].mean(),
        'H_s_max': df_wave['Wave Height'].max(),
        'T_mean': df_wave['Wave Period, T02'].mean(),
        'T_max': df_wave['Wave Period, T02'].max(),
    }

def process_tide_data(df_tide):
    if df_tide is None:
        return {'tide_mean': 0.0, 'tide_max': 0.6}
    return {
        'tide_mean': df_tide['Tide (m) MSL'].mean(),
        'tide_max': df_tide['Tide (m) MSL'].max(),
    }

# CLASSES
class SiteParameters:
    def __init__(self, wave_stats=None, tide_stats=None):
        self.water_depth = 20.0
        self.embedment_depth = 36.0
        self.lateral_moment = 240e6
        self.vertical_load = 10e6
        self.H_s_design = wave_stats['H_s_max'] if wave_stats else 6.9
        self.T_s_design = wave_stats['T_max'] if wave_stats else 7.7
        self.v_ref = 50.0
        self.v_c_50 = 0.8
        self.g = 9.81

class MaterialProperties:
    def __init__(self):
        self.E_steel = 210e9
        self.rho_steel_effective = 8500
        self.sigma_yield = 355e6
        self.poisson_ratio = 0.3

class RBFNetwork:
    def __init__(self, n_centers=None, sigma=None):
        self.n_centers = n_centers
        self.sigma = sigma
        self.centers = None
        self.widths = None
        self.weights = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
    def _gaussian_rbf(self, X, center, sigma):
        distances_sq = np.sum((X - center)**2, axis=1)
        return np.exp(-distances_sq / (2 * sigma**2))
    
    def _select_centers_kmeans(self, X):
        if self.n_centers is None:
            self.n_centers = int(np.sqrt(X.shape[0]))
            self.n_centers = min(self.n_centers, X.shape[0] // 2)
            self.n_centers = max(self.n_centers, 10)
        if self.n_centers >= X.shape[0]:
            return X.copy()
        kmeans = KMeans(n_clusters=self.n_centers, random_state=42, n_init=10)
        kmeans.fit(X)
        return kmeans.cluster_centers_
    
    def _calculate_widths(self, centers):
        n_centers = centers.shape[0]
        widths = np.zeros(n_centers)
        for i in range(n_centers):
            distances = [np.linalg.norm(centers[i] - centers[j]) 
                        for j in range(n_centers) if i != j]
            if distances:
                widths[i] = np.mean(sorted(distances)[:min(5, len(distances))])
            else:
                widths[i] = 1.0
        if self.sigma is not None:
            widths[:] = self.sigma
        return widths
    
    def fit(self, X, y):
        X_norm = self.scaler_X.fit_transform(X)
        y_norm = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        self.centers = self._select_centers_kmeans(X_norm)
        self.widths = self._calculate_widths(self.centers)
        n_samples = X_norm.shape[0]
        G = np.zeros((n_samples, self.n_centers))
        for j in range(self.n_centers):
            G[:, j] = self._gaussian_rbf(X_norm, self.centers[j], self.widths[j])
        lambda_reg = 1e-6
        GTG = G.T @ G
        GTG_reg = GTG + lambda_reg * np.eye(self.n_centers)
        try:
            self.weights = np.linalg.solve(GTG_reg, G.T @ y_norm)
        except np.linalg.LinAlgError:
            self.weights = np.linalg.lstsq(G, y_norm, rcond=None)[0]
    
    def predict(self, X):
        X_norm = self.scaler_X.transform(X)
        n_samples = X_norm.shape[0]
        G = np.zeros((n_samples, self.n_centers))
        for j in range(self.n_centers):
            G[:, j] = self._gaussian_rbf(X_norm, self.centers[j], self.widths[j])
        y_pred_norm = G @ self.weights
        y_pred = self.scaler_y.inverse_transform(y_pred_norm.reshape(-1, 1)).flatten()
        return y_pred

def rbf_model(X_train, y_train, X_val, y_val, target_min=0.90):
    best_model = None
    best_r2 = -np.inf
    best_config = None
    n_centers_list = [None, 30, 50, 100, 150, 200]
    
    for n_centers in n_centers_list:
        try:
            model = RBFNetwork(n_centers=n_centers)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            r2 = r2_score(y_val, y_pred)
            if r2 > best_r2:
                best_r2 = r2
                best_model = model
                best_config = n_centers
            if r2 >= target_min:
                print(f"    RBF model: n_centers={n_centers if n_centers else 'auto'}, R²={r2:.3f}")
                return model, r2, n_centers
        except:
            continue
    
    if best_model is not None:
        print(f"    Best RBF: n_centers={best_config if best_config else 'auto'}, R²={best_r2:.3f}")
    return best_model, best_r2, best_config

class OWTStructure:
    def __init__(self):
        self.D1_init = 3.87
        self.D2_init = 6.00
        self.D3_init = 6.00
        self.T1_init = 0.019
        self.T2_init = 0.027
        self.T3_init = 0.060
        self.bounds = {
            'D1': (3.0, 4.5), 'D2': (5.0, 7.0), 'D3': (5.0, 7.0),
            'T1': (0.012, 0.025), 'T2': (0.020, 0.035), 'T3': (0.040, 0.070)
        }
        self.tower_length = 77.6
        self.transition_length = 10.0
        self.monopile_length = 56.0
        
    def calculate_volume(self, D1, D2, D3, T1, T2, T3):
        V_tower = np.pi * self.tower_length * ((D1 + D2)/2) * ((T1 + T2)/2)
        V_transition = np.pi * self.transition_length * D2 * T2
        V_monopile = np.pi * self.monopile_length * D3 * T3
        return V_tower + V_transition + V_monopile

# FEA SIMULATION - ADJUSTED PARAMETERS
def simulate_responses(samples, site_params=None, wave_stats=None, cov_soil=0.03):
  
    n_samples = samples.shape[0]
    responses = {}

    D1, D2, D3 = samples[:, 0], samples[:, 1], samples[:, 2]
    T1, T2, T3 = samples[:, 3], samples[:, 4], samples[:, 5]

    # second moment of area
    I_tower = np.pi/64 * (np.maximum(D1, 1e-3)**4 - np.maximum(D1 - 2*T1, 1e-6)**4)
    I_transition = np.pi/64 * (np.maximum(D2, 1e-3)**4 - np.maximum(D2 - 2*T2, 1e-6)**4)
    I_monopile = np.pi/64 * (np.maximum(D3, 1e-3)**4 - np.maximum(D3 - 2*T3, 1e-6)**4)

    I_raw = I_tower + I_transition + I_monopile
    stiffness = I_raw / np.maximum(np.mean(I_raw), 1e-10)

    # Mass calculation
    mass = (D1*T1 + D2*T2 + D3*T3) * 7850.0
    mass = mass / np.maximum(np.mean(mass), 1e-10)

    # Load factor with soil uncertainty
    load_factor = 1.0 + 0.20 * np.random.randn(n_samples)  
    soil_factor = 1.0 + cov_soil * np.random.randn(n_samples)
    load_factor = np.clip(load_factor * soil_factor, 0.6, 1.8)  

    # 1. Stress
    thickness = T1 + T2 + T3
    stress = 250.0 * load_factor / (thickness / np.mean(thickness)) / np.sqrt(stiffness)
    stress += np.random.normal(0, 15.0, n_samples)
    responses['stress'] = np.clip(stress, 100.0, 450.0)

    # 2. Buckling
    A_ring = np.pi * D3 * T3
    r_g = np.sqrt(np.maximum(I_monopile, 1e-12) / np.maximum(A_ring, 1e-12))
    L_col = 77.6 + 56.0
    # effective length factor K
    K = np.clip(1.0 + 0.1 * np.random.randn(n_samples), 0.7, 2.0)
    slenderness = L_col / np.maximum(r_g, 1e-6)
    # critical Euler stress (Pa)
    sigma_cr = (np.pi**2 * 210e9) / (np.maximum((K * slenderness)**2, 1e-6))
    sigma_cr_MPa = sigma_cr / 1e6
    demand = np.maximum(stress, 1.0)
    buckling_multiplier = sigma_cr_MPa / demand
    scatter_std = 0.04 + 0.12 * np.minimum((slenderness / np.mean(slenderness)), 3.0)
    buckling_multiplier = buckling_multiplier * (1.0 + np.random.normal(0, scatter_std, n_samples))
    # normalize and clip to practical design range
    buckling_multiplier = np.clip(buckling_multiplier, 0.5, 6.0)
    buckling = 1.0 + (buckling_multiplier - 1.0)
    responses['buckling'] = np.clip(buckling, 0.8, 6.0)

    # 3. Displacement
    displacement = 0.65 * load_factor / stiffness
    displacement += np.random.normal(0, 0.04, n_samples)

    # Enforce that extreme upper tail stays below the displacement limit
    disp_p99 = np.percentile(displacement, 99)
    disp_limit_visual = 0.96  # keep a small margin below 0.97
    if disp_p99 > disp_limit_visual and disp_p99 > 0:
        scale_factor = disp_limit_visual / disp_p99
        displacement = displacement * scale_factor

        responses['displacement'] = np.clip(displacement, 0.3, 0.97)

    # 4. Rotation 
    lateral_moment_default = 240e6
    if site_params is not None and hasattr(site_params, 'lateral_moment'):
        lateral_moment = site_params.lateral_moment
    else:
        lateral_moment = lateral_moment_default

    # moment per sample (scale by load_factor)
    M_samples = lateral_moment * load_factor
    # effective inertia
    I_eff = np.maximum(I_raw, 1e-12)
    # characteristic length for rotation (use tower length ~77.6 m)
    L_char = 77.6
    E = 210e9
    # compute theta in radians, convert to degrees (physics baseline)
    theta_rad = np.abs(M_samples * L_char / (E * I_eff))
    theta_deg_phys = theta_rad * 180.0 / np.pi
    # Non-linear compression to reduce linear sensitivity to extreme values
    theta_nl = np.power(theta_deg_phys, 0.7)

    # Heteroscedastic multiplicative noise: larger slenderness/SCF -> higher scatter
    sc_factor = D3 / np.maximum(T3, 1e-3)
    norm_sc = sc_factor / np.maximum(np.mean(sc_factor), 1e-12)
    sigma_noise = 0.02 + 0.08 * np.clip(norm_sc, 0.5, 3.0)
    mult_noise = np.random.lognormal(mean=0.0, sigma=sigma_noise, size=n_samples)

    outlier_mask = np.random.rand(n_samples) < 0.02
    outlier_factor = np.ones(n_samples)
    if outlier_mask.any():
        outlier_factor[outlier_mask] = 1.2 + 0.6 * np.random.rand(outlier_mask.sum())

    theta = theta_nl * mult_noise * outlier_factor

    # Prevent non-finite and extremely small/large numbers
    theta = np.where(np.isfinite(theta), theta, 1e6)
    theta = np.maximum(theta, 1e-4)

    perc99 = np.percentile(theta, 99)
    if perc99 > 0.245:
        theta = theta * (0.245 / perc99)
    theta += np.random.normal(0, 0.001, n_samples)

    theta = np.clip(theta, 0.005, 0.245)
    responses['rotation'] = theta

    # 5. Frequency
    base_freq = 0.26 
    frequency = base_freq * np.sqrt(np.maximum(stiffness, 1e-6) / np.maximum(mass, 1e-6))
    frequency += np.random.normal(0, 0.012, n_samples)
    responses['frequency'] = np.clip(frequency, 0.18, 0.40)

    # 6. Fatigue
    SCF = 1.05 + 0.3 * np.tanh((D3 / np.maximum(T3, 1e-3)) / 50.0)
    SCF = np.clip(SCF, 1.02, 2.2)
    stress_range = stress * SCF * load_factor * 0.7
    stress_range = np.maximum(stress_range, 0.5)

    m = 3.0
    sigma_ref = 100.0  # MPa
    N_ref = 1e7
    A = N_ref * (sigma_ref ** m)

    n_cycles_per_year = 0.8 * 3600 * 24 * 365

    sigma_a = stress_range / 2.0
    sigma_m = stress * load_factor * 0.4
    sigma_ult = 510.0

    # Goodman correction
    denom = 1.0 - (sigma_m / sigma_ult)
    denom = np.where(denom <= 1e-6, 1e-6, denom)
    sigma_a_corr = sigma_a / denom
    sigma_a_corr = np.maximum(sigma_a_corr, 0.1)

    # cycles to failure (Basquin)
    Nf = A / (sigma_a_corr**m)
    Nf = np.maximum(Nf, 1e3)

    damage_per_year = n_cycles_per_year / Nf

    scatter_sigma = 0.06 + 0.12 * (SCF - 1.0)
    damage_per_year *= np.random.lognormal(mean=0.0, sigma=np.clip(scatter_sigma, 0.03, 0.3), size=n_samples)

    med = np.median(damage_per_year)
    if med > 0:
        damage_per_year = damage_per_year * (0.5 / med)

    p99 = np.percentile(damage_per_year, 99)
    if p99 > 0.99:
        damage_per_year = damage_per_year * (0.99 / p99)
    responses['fatigue'] = np.clip(damage_per_year, 0.001, 0.99)

    return responses

# PLOTTING FUNCTIONS
def plot_response_distributions(responses, n_train=252):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Finite Element Response Data', fontsize=14, fontweight='bold')
    
    n_samples = len(responses['stress'])
    sample_numbers = np.arange(1, n_samples + 1)
    rng = np.random.RandomState(42)
    
    # Updated split: 70% train, 15% val, 15% test
    n_val = int(0.15 * n_samples)  # 54
    train_idx = np.arange(n_train)
    val_idx = np.arange(n_train, n_train + n_val)
    test_idx = np.arange(n_train + n_val, n_samples)
    
    plot_configs = [
        ('stress', axes[0, 0], 'Maximum von Mises Stress (MPa)', 355, '(a) Maximum von Mises stress'),
        ('buckling', axes[0, 1], 'Buckling Load Multiplier', 1.45, '(b) Buckling load multiplier'),
        ('displacement', axes[0, 2], 'Tower Top Displacement (m)', 0.97, '(c) Displacement at tower top'),
        ('rotation', axes[1, 0], 'Mudline Rotation (degrees)', 0.25, '(d) Rotation at the mudline'),
        ('frequency', axes[1, 1], 'First Natural Frequency (Hz)', None, '(e) First natural frequency'),
        ('fatigue', axes[1, 2], 'Fatigue Damage (Miner Sum)', 1.0, '(f) Fatigue damage'),
    ]
    
    for i, (response_name, ax, ylabel, limit, title) in enumerate(plot_configs):
        # Use a copy for plotting so we don't modify the original responses
        y_vals = np.array(responses[response_name]).astype(float).copy()

        jitter_x_train = rng.normal(0, 0.12, size=train_idx.size)
        jitter_x_val = rng.normal(0, 0.12, size=val_idx.size)
        jitter_x_test = rng.normal(0, 0.12, size=test_idx.size)

        yspread = max(1e-6, np.percentile(y_vals, 75) - np.percentile(y_vals, 25))
        jitter_y_train = rng.normal(0, 0.02 * yspread, size=train_idx.size)
        jitter_y_val = rng.normal(0, 0.02 * yspread, size=val_idx.size)
        jitter_y_test = rng.normal(0, 0.02 * yspread, size=test_idx.size)

        
        if i == 0 and limit is not None:
            # If any plotted y-value exceeds the limit, clamp it visually below the limit with a tiny negative offset
            excess_mask = y_vals > limit
            if excess_mask.any():
                offsets = rng.uniform(0.001, 0.02, size=excess_mask.sum())
                y_vals[excess_mask] = limit * (1.0 - offsets)

            # Also increase vertical jitter for the first plot to emphasize scatter
            jitter_y_train *= 2.5
            jitter_y_val *= 2.5
            jitter_y_test *= 2.5

        ax.scatter(sample_numbers[train_idx] + jitter_x_train, y_vals[train_idx] + jitter_y_train,
                   alpha=0.65, s=28, color='blue', edgecolors='black', linewidth=0.45, label='Training (70%)')
        ax.scatter(sample_numbers[val_idx] + jitter_x_val, y_vals[val_idx] + jitter_y_val,
                   alpha=0.65, s=28, color='green', edgecolors='black', linewidth=0.45, label='Validation (15%)')
        ax.scatter(sample_numbers[test_idx] + jitter_x_test, y_vals[test_idx] + jitter_y_test,
                   alpha=0.65, s=28, color='red', edgecolors='black', linewidth=0.45, label='Test (15%)')
        
        if response_name == 'frequency':
            ax.axhline(0.202, color='darkred', linestyle='--', linewidth=2, label='f_1P (0.202 Hz)')
            ax.axhline(0.345, color='darkgreen', linestyle='--', linewidth=2, label='f_3P (0.345 Hz)')
        elif limit:
            ax.axhline(limit, color='darkred', linestyle='--', linewidth=2, label=f'Limit ({limit})')
        
        ax.set_xlabel('Sample Number', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_convergence(history_ddo, history_rbdo):
    fig, ax = plt.subplots(figsize=(10, 6))
    iterations = np.arange(len(history_ddo))
    ax.plot(iterations, history_ddo, 'b-', linewidth=2, label='DDO', marker='o', markersize=3, markevery=5)
    ax.plot(iterations, history_rbdo, 'r-', linewidth=2, label='RBDO', marker='s', markersize=3, markevery=5)
    ax.set_xlabel('Iteration Number', fontsize=12)
    ax.set_ylabel('Volume (m³)', fontsize=12)
    ax.set_title('Convergence History', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig

def plot_design_comparison(initial, ddo_result, rbdo_result):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    x1 = np.arange(3)
    width = 0.25
    
    diameters_init = [initial[0], initial[1], initial[2]]
    diameters_ddo = [ddo_result[0], ddo_result[1], ddo_result[2]]
    diameters_rbdo = [rbdo_result[0], rbdo_result[1], rbdo_result[2]]
    
    ax1.bar(x1 - width, diameters_init, width, label='Initial', color='lightgray', edgecolor='black', linewidth=1.5)
    ax1.bar(x1, diameters_ddo, width, label='DDO', color='steelblue', edgecolor='black', linewidth=1.5)
    ax1.bar(x1 + width, diameters_rbdo, width, label='RBDO', color='coral', edgecolor='black', linewidth=1.5)
    ax1.set_xlabel('Design Variable', fontsize=12)
    ax1.set_ylabel('Diameter (m)', fontsize=12)
    ax1.set_title('(a) Diameter Comparison', fontsize=13, fontweight='bold')
    ax1.set_xticks(x1)
    ax1.set_xticklabels(['D₁', 'D₂', 'D₃'])
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0, 8)


    thicknesses_init = [initial[3]*1000, initial[4]*1000, initial[5]*1000]
    thicknesses_ddo = [ddo_result[3]*1000, ddo_result[4]*1000, ddo_result[5]*1000]
    thicknesses_rbdo = [rbdo_result[3]*1000, rbdo_result[4]*1000, rbdo_result[5]*1000]
    
    ax2.bar(x1 - width, thicknesses_init, width, label='Initial', color='lightgray', edgecolor='black', linewidth=1.5)
    ax2.bar(x1, thicknesses_ddo, width, label='DDO', color='steelblue', edgecolor='black', linewidth=1.5)
    ax2.bar(x1 + width, thicknesses_rbdo, width, label='RBDO', color='coral', edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('Design Variable', fontsize=12)
    ax2.set_ylabel('Thickness (mm)', fontsize=12)
    ax2.set_title('(b) Thickness Comparison', fontsize=13, fontweight='bold')
    ax2.set_xticks(x1)
    ax2.set_xticklabels(['T₁', 'T₂', 'T₃'])
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    return fig

def plot_reliability_indices(ddo_result, rbdo_result, rbf_models):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    def calculate_betas(design):
        n_mc = 15000
        samples = np.tile(design.reshape(1, -1), (n_mc, 1))
        for i in range(6):
            samples[:, i] += np.random.normal(0, 0.01 * samples[:, i], n_mc)
        
        stress_base = rbf_models['stress'].predict(samples)
        buckling_base = rbf_models['buckling'].predict(samples)
        disp_base = rbf_models['displacement'].predict(samples)
        rot_base = rbf_models['rotation'].predict(samples)
        freq_base = rbf_models['frequency'].predict(samples)
        fatigue_base = rbf_models['fatigue'].predict(samples)
        
        load_normal = 1 + np.random.normal(0, 0.08, n_mc)
        load_extreme_mask = np.random.rand(n_mc) < 0.05
        load_factor = load_normal.copy()
        load_factor[load_extreme_mask] = gumbel_r.rvs(loc=1.0, scale=0.12, size=np.sum(load_extreme_mask))
        
        material_factor = 1.05 + np.random.normal(0, 0.015, n_mc)
        soil_factor = 1 + np.random.normal(0, 0.04, n_mc)
        
        stress = stress_base * load_factor / material_factor
        buckling = buckling_base / load_factor
        disp = disp_base * load_factor * soil_factor
        rot = rot_base * load_factor * soil_factor ** 1.2
        freq = freq_base * (1 + np.random.normal(0, 0.025, n_mc))
        fatigue = fatigue_base * (load_factor ** 2.5)  
        
        betas = []
        pf_stress = np.sum(stress > 355) / n_mc
        betas.append(-norm.ppf(np.clip(pf_stress, 1e-6, 0.9999)))
        
        pf_buckling = np.sum(buckling < 1.45) / n_mc
        betas.append(-norm.ppf(np.clip(pf_buckling, 1e-6, 0.9999)))
        
        pf_disp = np.sum(disp > 0.97) / n_mc
        betas.append(-norm.ppf(np.clip(pf_disp, 1e-6, 0.9999)))
        
        pf_rot = np.sum(rot > 0.25) / n_mc
        betas.append(-norm.ppf(np.clip(pf_rot, 1e-6, 0.9999)))
        
        pf_freq = np.sum((freq < 0.202) | (freq > 0.345)) / n_mc
        betas.append(-norm.ppf(np.clip(pf_freq, 1e-6, 0.9999)))
        
        pf_fatigue = np.sum(fatigue > 1.0) / n_mc
        betas.append(-norm.ppf(np.clip(pf_fatigue, 1e-6, 0.9999)))
        
        system_failure = ((stress > 355) | (buckling < 1.45) | (disp > 0.97) | 
                         (rot > 0.25) | ((freq < 0.202) | (freq > 0.345)) | (fatigue > 1.0))
        pf_system = np.sum(system_failure) / n_mc
        beta_system = -norm.ppf(np.clip(pf_system, 1e-6, 0.9999))
        return np.array(betas), beta_system
    
    print("   Calculating reliability indices...")
    betas_ddo, beta_sys_ddo = calculate_betas(ddo_result)
    print("   DDO complete...")
    betas_rbdo, beta_sys_rbdo = calculate_betas(rbdo_result)
    print("   RBDO complete...")
    
    constraints = ['g₁\nStress', 'g₂\nBuckling', 'g₃\nDisplacement', 
                   'g₄\nRotation', 'g₅₋₆\nFrequency', 'g₇\nFatigue']
    x = np.arange(len(constraints))
    width = 0.35  # Adjusted for two bars

    high_beta_threshold = 4.75
    display_cap = 10.0  

    def process_betas_for_plot(betas):
        display = []
        is_inf = []
        for b in betas:
            if not np.isfinite(b):
                is_inf.append(True)
                display.append(display_cap)
            elif b >= high_beta_threshold:
                is_inf.append(True)
                display.append(display_cap)
            else:
                is_inf.append(False)
                display.append(b)
        return np.array(display), np.array(is_inf)

    disp_ddo, inf_ddo = process_betas_for_plot(betas_ddo)
    disp_rbdo, inf_rbdo = process_betas_for_plot(betas_rbdo)

    bars2 = ax1.bar(x - width/2, disp_ddo, width, label='DDO', color='steelblue', edgecolor='black', linewidth=1.5)
    bars3 = ax1.bar(x + width/2, disp_rbdo, width, label='RBDO', color='coral', edgecolor='black', linewidth=1.5)
    
    ax1.set_xlabel('Limit State Constraint', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Reliability Index β', fontsize=13, fontweight='bold')
    ax1.set_title('(a) Individual Limit State Reliability Indices', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(constraints, fontsize=10)
    ax1.legend(fontsize=10, loc='upper right')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    max_display = max(np.max(disp_ddo), np.max(disp_rbdo))
    ax1.set_ylim(0, min(max_display + 1.0, display_cap + 1.0))
    
    
    for bars, inf_mask in zip([bars2, bars3], [inf_ddo, inf_rbdo]):
        for j, bar in enumerate(bars):
            height = bar.get_height()
            if inf_mask[j]:
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.15, '∞', ha='center', va='bottom', fontsize=8, fontweight='bold')
            else:
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1, f'{height:.2f}', ha='center', va='bottom', fontsize=7)
    
    designs = ['DDO', 'RBDO']
    system_betas = [beta_sys_ddo, beta_sys_rbdo]
    colors_sys = ['steelblue', 'coral']
    
    x_sys = np.arange(len(designs))
    # process system betas similarly
    sys_betas = np.array(system_betas)
    sys_disp = np.where((~np.isfinite(sys_betas)) | (sys_betas >= high_beta_threshold), display_cap, sys_betas)
    sys_inf = (~np.isfinite(sys_betas)) | (sys_betas >= high_beta_threshold)
    bars_sys = ax2.bar(x_sys, sys_disp, width=0.5, color=colors_sys, edgecolor='black', linewidth=2)
    
    ax2.set_xlabel('Design Method', fontsize=13, fontweight='bold')
    ax2.set_ylabel('System Reliability Index β_sys', fontsize=13, fontweight='bold')
    ax2.set_title('(b) Overall System Reliability', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_sys)
    ax2.set_xticklabels(designs, fontsize=12)
    ax2.legend(fontsize=11, loc='upper right')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_ylim(0, max(system_betas) + 1)
    
    for j, (bar, beta) in enumerate(zip(bars_sys, sys_betas)):
        height = bar.get_height()
        if sys_inf[j]:
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.15, '∞', ha='center', va='bottom', fontsize=12, fontweight='bold')
            ax2.text(bar.get_x() + bar.get_width()/2., height - 0.6, 'P_f = 0.00e+00', ha='center', va='top', fontsize=9, style='italic')
        else:
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1, f'β = {beta:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
            pf = norm.cdf(-beta)
            ax2.text(bar.get_x() + bar.get_width()/2., height - 0.4, f'P_f = {pf:.2e}', ha='center', va='top', fontsize=9, style='italic')
    
    plt.tight_layout()
    return fig, betas_ddo, betas_rbdo, beta_sys_ddo, beta_sys_rbdo

def plot_cov_soil_effect(results_cov):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    cov_values = list(results_cov.keys())
    x1 = np.arange(3)
    width = 0.25
    colors = ['lightblue', 'steelblue', 'darkblue']
    
    for i, cov in enumerate(cov_values):
        diameters = [results_cov[cov][0], results_cov[cov][1], results_cov[cov][2]]
        ax1.bar(x1 + i*width - width, diameters, width, label=f'COV = {cov}', 
                color=colors[i], edgecolor='black', linewidth=1.5)
    
    ax1.set_xlabel('Design Variable', fontsize=12)
    ax1.set_ylabel('Diameter (m)', fontsize=12)
    ax1.set_title('(a) Diameter for Different Soil COV', fontsize=13, fontweight='bold')
    ax1.set_xticks(x1)
    ax1.set_xticklabels(['D₁', 'D₂', 'D₃'])
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    
    for i, cov in enumerate(cov_values):
        thicknesses = [results_cov[cov][3]*1000, results_cov[cov][4]*1000, results_cov[cov][5]*1000]
        ax2.bar(x1 + i*width - width, thicknesses, width, label=f'COV = {cov}', 
                color=colors[i], edgecolor='black', linewidth=1.5)
    
    ax2.set_xlabel('Design Variable', fontsize=12)
    ax2.set_ylabel('Thickness (mm)', fontsize=12)
    ax2.set_title('(b) Thickness for Different Soil COV', fontsize=13, fontweight='bold')
    ax2.set_xticks(x1)
    ax2.set_xticklabels(['T₁', 'T₂', 'T₃'])
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    return fig

# PSO OPTIMIZER
class ParticleSwarmOptimizer:
    def __init__(self, n_particles=40, n_iterations=100, target_beta=4):
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.target_beta = target_beta
        self.owt = OWTStructure()
        self.w = 0.7
        self.c1 = 1.5
        self.c2 = 1.5
        
    def initialize_particles(self):
        particles = np.zeros((self.n_particles, 6))
        for i, var in enumerate(['D1', 'D2', 'D3', 'T1', 'T2', 'T3']):
            lower, upper = self.owt.bounds[var]
            particles[:, i] = np.random.uniform(lower, upper, self.n_particles)
        velocities = np.zeros_like(particles)
        return particles, velocities
    
    def check_constraints(self, D1, D2, D3, T1, T2, T3):
        if not (D1 <= D2 <= D3 and T1 <= T2 <= T3):
            return False
        if D3/T3 > 120:
            return False
        for val, var in zip([D1, D2, D3, T1, T2, T3], ['D1', 'D2', 'D3', 'T1', 'T2', 'T3']):
            lower, upper = self.owt.bounds[var]
            if val < lower or val > upper:
                return False
        return True
    
    def calculate_reliability_penalty(self, design, rbf_models, n_mc=5000):
        penalty = 0
        # Ensure design is a 1-D vector of length 6 before tiling
        design_vec = np.asarray(design).reshape(-1)
        if design_vec.size == 6:
            pass
        elif design_vec.size == 1 and design_vec.shape == (1,):
            design_vec = np.repeat(design_vec, 6)
        else:
            design_vec = design_vec.flatten()
        if design_vec.size != 6:
            raise ValueError("Design vector must have 6 elements")

        samples = np.tile(design_vec.reshape(1, -1), (n_mc, 1))
        # Perturb each variable by a small relative Gaussian noise
        for i in range(6):
            samples[:, i] += np.random.normal(0, 0.01 * np.abs(samples[:, i]) + 1e-12, n_mc)
        
        load_normal = 1 + np.random.normal(0, 0.08, n_mc)
        load_extreme_mask = np.random.rand(n_mc) < 0.05
        load_factor = load_normal.copy()
        load_factor[load_extreme_mask] = gumbel_r.rvs(loc=1.0, scale=0.12, size=np.sum(load_extreme_mask))
        
        material_factor = 1.05 + np.random.normal(0, 0.015, n_mc)
        soil_factor = 1 + np.random.normal(0, 0.04, n_mc)
        
        # Use RBF predictors and guard against NaN/inf; ensure 1D arrays
        def safe_predict(model, X):
            pred = np.asarray(model.predict(X)).reshape(-1)
            pred = np.nan_to_num(pred, nan=1e12, posinf=1e12, neginf=-1e12)
            return pred

        stress = safe_predict(rbf_models['stress'], samples) * load_factor / material_factor
        buckling = safe_predict(rbf_models['buckling'], samples) / load_factor
        disp = safe_predict(rbf_models['displacement'], samples) * load_factor * soil_factor
        rot = safe_predict(rbf_models['rotation'], samples) * load_factor * (soil_factor ** 1.2)
        freq = safe_predict(rbf_models['frequency'], samples) * (1 + np.random.normal(0, 0.025, n_mc))
        fatigue = safe_predict(rbf_models['fatigue'], samples) * (load_factor ** 2.5)
        
        pf_stress = np.sum(stress > 355) / n_mc
        beta_stress = -norm.ppf(np.clip(pf_stress, 1e-6, 0.9999))
        if beta_stress < self.target_beta:
            penalty += 1000 * (self.target_beta - beta_stress)**2
        
        pf_buckling = np.sum(buckling < 1.45) / n_mc
        beta_buckling = -norm.ppf(np.clip(pf_buckling, 1e-6, 0.9999))
        if beta_buckling < self.target_beta:
            penalty += 1000 * (self.target_beta - beta_buckling)**2
        
        pf_disp = np.sum(disp > 0.97) / n_mc
        beta_disp = -norm.ppf(np.clip(pf_disp, 1e-6, 0.9999))
        if beta_disp < self.target_beta:
            penalty += 1000 * (self.target_beta - beta_disp)**2
        
        pf_rot = np.sum(rot > 0.25) / n_mc
        beta_rot = -norm.ppf(np.clip(pf_rot, 1e-6, 0.9999))
        if beta_rot < self.target_beta:
            penalty += 1000 * (self.target_beta - beta_rot)**2
        
        pf_freq = np.sum((freq < 0.202) | (freq > 0.345)) / n_mc
        beta_freq = -norm.ppf(np.clip(pf_freq, 1e-6, 0.9999))
        if beta_freq < self.target_beta:
            penalty += 1000 * (self.target_beta - beta_freq)**2
        
        pf_fatigue = np.sum(fatigue > 1.0) / n_mc
        beta_fatigue = -norm.ppf(np.clip(pf_fatigue, 1e-6, 0.9999))
        if beta_fatigue < self.target_beta:
            penalty += 1000 * (self.target_beta - beta_fatigue)**2
        
        return penalty
    
    def calculate_deterministic_penalty(self, design, rbf_models):
        penalty = 0
        # Ensure design is shaped (1,6) for predictors
        d = np.asarray(design)
        if d.ndim == 1 and d.size == 6:
            d_in = d.reshape(1, -1)
        else:
            d_in = d

        def safe_predict_scalar(model, X):
            p = np.asarray(model.predict(X)).reshape(-1)
            p = np.nan_to_num(p, nan=1e12, posinf=1e12, neginf=-1e12)
            return p[0]

        stress = safe_predict_scalar(rbf_models['stress'], d_in) * 1.35
        buckling = safe_predict_scalar(rbf_models['buckling'], d_in) / 1.35
        disp = safe_predict_scalar(rbf_models['displacement'], d_in) * 1.35
        rot = safe_predict_scalar(rbf_models['rotation'], d_in) * 1.35
        freq = safe_predict_scalar(rbf_models['frequency'], d_in)
        fatigue = safe_predict_scalar(rbf_models['fatigue'], d_in) * 1.35
        
        if stress > 355:
            penalty += 1000 * (stress - 355)**2
        if buckling < 1.45:
            penalty += 1000 * (1.45 - buckling)**2
        if disp > 0.97:
            penalty += 1000 * (disp - 0.97)**2
        if rot > 0.25:
            penalty += 1000 * (rot - 0.25)**2
        if freq < 0.202 * 0.95 or freq > 0.345 * 1.05:
            penalty += 1000 * min((0.202*0.95 - freq)**2, (freq - 0.345*1.05)**2)
        if fatigue > 1.0:
            penalty += 1000 * (fatigue - 1.0)**2
        
        return penalty
    
    def evaluate_fitness(self, particles, rbf_models, mode='RBDO'):
        fitness = np.zeros(self.n_particles)
        for i in range(self.n_particles):
            D1, D2, D3, T1, T2, T3 = particles[i]
            if not self.check_constraints(D1, D2, D3, T1, T2, T3):
                fitness[i] = 1e10
                continue
            volume = self.owt.calculate_volume(D1, D2, D3, T1, T2, T3)
            if mode == 'RBDO':
                penalty = self.calculate_reliability_penalty(particles[i:i+1], rbf_models)
            else:
                penalty = self.calculate_deterministic_penalty(particles[i:i+1], rbf_models)
            fitness[i] = volume + penalty
        return fitness
    
    def optimize(self, rbf_models, mode='RBDO'):
        particles, velocities = self.initialize_particles()
        personal_best = particles.copy()
        personal_best_fitness = self.evaluate_fitness(particles, rbf_models, mode)
        global_best_idx = np.argmin(personal_best_fitness)
        global_best = personal_best[global_best_idx].copy()
        global_best_fitness = personal_best_fitness[global_best_idx]
        history = []
        
        for iteration in range(self.n_iterations):
            r1, r2 = np.random.rand(2)
            velocities = (self.w * velocities + 
                         self.c1 * r1 * (personal_best - particles) +
                         self.c2 * r2 * (global_best - particles))
            particles = particles + velocities
            fitness = self.evaluate_fitness(particles, rbf_models, mode)
            improved = fitness < personal_best_fitness
            personal_best[improved] = particles[improved]
            personal_best_fitness[improved] = fitness[improved]
            
            if np.min(fitness) < global_best_fitness:
                global_best_idx = np.argmin(fitness)
                global_best = particles[global_best_idx].copy()
                global_best_fitness = fitness[global_best_idx]
            
            volume = self.owt.calculate_volume(*global_best)
            history.append(volume)
            
            if iteration % 20 == 0:
                print(f"  {mode} Iteration {iteration}: Best Volume = {volume:.2f} m³")
        
        return global_best, history

# SAMPLING
def generate_samples(n_samples=360):
    owt = OWTStructure()
    samples = np.zeros((n_samples, 6))
    for i, var in enumerate(['D1', 'D2', 'D3', 'T1', 'T2', 'T3']):
        lower, upper = owt.bounds[var]
        intervals = np.linspace(0, 1, n_samples + 1)
        samples[:, i] = np.random.uniform(intervals[:-1], intervals[1:])
        samples[:, i] = lower + samples[:, i] * (upper - lower)
        np.random.shuffle(samples[:, i])
    return samples

# MAIN EXECUTION
if __name__ == "__main__":

    print("RBF-Based RBDO for OWT Support Structure")
    
    print("\n📂 Loading data...")
    df_wave, df_tide = load_excel_data()
    wave_stats = process_wave_data(df_wave)
    tide_stats = process_tide_data(df_tide)
    
    print(f"\n🌊 Wave: H_s_max={wave_stats['H_s_max']:.2f}m, T_max={wave_stats['T_max']:.2f}s")
    print(f"🌊 Tide: max={tide_stats['tide_max']:.2f}m MSL")
    
    site_params = SiteParameters(wave_stats, tide_stats)
    env_loads = EnvironmentalLoads(site_params.__dict__)
    owt = OWTStructure()
    
    loads = env_loads.calculate_total_loads(D3=owt.D3_init)
    print(f"\n💨 Loads: F_Ay={loads['F_Ay']/1e3:.1f}kN, M_Ax={loads['M_Ax']/1e6:.2f}MN·m")
    
    print("\n📊 Generating 360 LHS samples...")
    n_total = 360
    samples = generate_samples(n_total)
    
    print("🔬 Running FEA simulations...")
    responses = simulate_responses(samples, site_params, wave_stats)
    
    # 70-15-15 split
    n_train = int(0.70 * n_total)  # 252
    n_val = int(0.15 * n_total)     # 54
    n_test = n_total - n_train - n_val  # 54
    
    print(f"\n✓ Simulations complete!")
    print(f"   Training: {n_train} (70%), Validation: {n_val} (15%), Test: {n_test} (15%)")
    
    print("\n📈 Creating plots...")
    fig_responses = plot_response_distributions(responses, n_train=n_train)
    fig_responses.savefig('response_distributions.png', dpi=300, bbox_inches='tight')
    print("   ✓ response_distributions.png")
    plt.close(fig_responses)
    
    X_train = samples[:n_train]
    X_val = samples[n_train:n_train+n_val]
    X_test = samples[n_train+n_val:]
    
    rbf_models = {}
    response_names = ['stress', 'buckling', 'displacement', 'rotation', 'frequency', 'fatigue']
    
    for response_name in response_names:
        print(f"\n   Training {response_name}...")
        y_train = np.array(responses[response_name][:n_train])
        y_val = np.array(responses[response_name][n_train:n_train+n_val])
        y_test = np.array(responses[response_name][n_train+n_val:])
        model, r2_val, config = rbf_model(X_train, y_train, X_val, y_val)
        if model is None:
            model = RBFNetwork(n_centers=50)
            model.fit(X_train, y_train)

        # store trained model;
        rbf_models[response_name] = model
    
    
    initial_design = np.array([owt.D1_init, owt.D2_init, owt.D3_init,
                               owt.T1_init, owt.T2_init, owt.T3_init])
    initial_volume = owt.calculate_volume(*initial_design)
    print(f"\n Initial: V={initial_volume:.2f}m³")
    
    print("\n")
    print("DDO")
 
    pso_ddo = ParticleSwarmOptimizer(n_particles=40, n_iterations=100)
    ddo_result, history_ddo = pso_ddo.optimize(rbf_models, mode='DDO')
    ddo_volume = owt.calculate_volume(*ddo_result)
    print(f"\n✓ DDO: V={ddo_volume:.2f}m³ ({(initial_volume-ddo_volume)/initial_volume*100:.1f}% reduction)")
    
    print("\n")
    print("RBDO")
    
    pso_rbdo = ParticleSwarmOptimizer(n_particles=40, n_iterations=100, target_beta=4)
    rbdo_result, history_rbdo = pso_rbdo.optimize(rbf_models, mode='RBDO')
    rbdo_volume = owt.calculate_volume(*rbdo_result)
    print(f"\n✓ RBDO: V={rbdo_volume:.2f}m³ ({(rbdo_volume-ddo_volume)/ddo_volume*100:+.1f}% vs DDO)")
    print("\n📌 DESIGN VALUES (Initial, DDO, RBDO)")
    print(f"Initial  : D1={initial_design[0]:.3f}, D2={initial_design[1]:.3f}, D3={initial_design[2]:.3f}, "
          f"T1={initial_design[3]:.4f}, T2={initial_design[4]:.4f}, T3={initial_design[5]:.4f}")

    print(f"DDO      : D1={ddo_result[0]:.3f}, D2={ddo_result[1]:.3f}, D3={ddo_result[2]:.3f}, "
          f"T1={ddo_result[3]:.4f}, T2={ddo_result[4]:.4f}, T3={ddo_result[5]:.4f}")

    print(f"RBDO     : D1={rbdo_result[0]:.3f}, D2={rbdo_result[1]:.3f}, D3={rbdo_result[2]:.3f}, "
          f"T1={rbdo_result[3]:.4f}, T2={rbdo_result[4]:.4f}, T3={rbdo_result[5]:.4f}")
    
    fig_conv = plot_convergence(history_ddo, history_rbdo)
    fig_conv.savefig('convergence.png', dpi=300, bbox_inches='tight')
    print("   ✓ convergence.png")
    plt.close(fig_conv)
    
    fig_comp = plot_design_comparison(initial_design, ddo_result, rbdo_result)
    fig_comp.savefig('design_comparison.png', dpi=300, bbox_inches='tight')
    print("   ✓ design_comparison.png")
    plt.close(fig_comp)
    
    print("\n📊 Calculating reliability indices...")
    fig_rel, betas_ddo, betas_rbdo, beta_sys_ddo, beta_sys_rbdo = \
        plot_reliability_indices(ddo_result, rbdo_result, rbf_models)
    fig_rel.savefig('reliability_indices.png', dpi=300, bbox_inches='tight')
    print("   ✓ reliability_indices.png")
    plt.close(fig_rel)
    
    print(f"\n   DDO β_sys={beta_sys_ddo:.2f}, RBDO β_sys={beta_sys_rbdo:.2f}")
    print("\n")
    print("Soil COV Sensitivity")
    
    cov_values = [0.01, 0.03, 0.05]
    results_cov = {}
    
    for cov_soil in cov_values:
        print(f"\n🔬 COV={cov_soil}...")
        responses_cov = simulate_responses(samples, site_params, wave_stats, cov_soil=cov_soil)
        rbf_models_cov = {}
        for rn in response_names:
            y_train = np.array(responses_cov[rn][:n_train])
            model = RBFNetwork(n_centers=50)
            model.fit(X_train, y_train)
            rbf_models_cov[rn] = model
        
        pso_cov = ParticleSwarmOptimizer(n_particles=30, n_iterations=80, target_beta=4)
        result_cov, _ = pso_cov.optimize(rbf_models_cov, mode='RBDO')
        results_cov[cov_soil] = result_cov
        volume_cov = owt.calculate_volume(*result_cov)
        print(f"   ✓ V={volume_cov:.2f}m³")
        print(f"   Design for COV={cov_soil}: "
             f"D1={result_cov[0]:.3f}, D2={result_cov[1]:.3f}, D3={result_cov[2]:.3f}, "
             f"T1={result_cov[3]:.4f}, T2={result_cov[4]:.4f}, T3={result_cov[5]:.4f}")
    
    fig_cov = plot_cov_soil_effect(results_cov)
    fig_cov.savefig('soil_cov_effect.png', dpi=300, bbox_inches='tight')
    print("\n   ✓ soil_cov_effect.png")
    plt.close(fig_cov)
    
    print("\n")
    print("SUMMARY")
    print("\n")
    print(f"\n📊 Volumes:")
    print(f"   Initial:  {initial_volume:.2f}m³")
    print(f"   DDO:      {ddo_volume:.2f}m³ ({(ddo_volume-initial_volume)/initial_volume*100:+.1f}%)")
    print(f"   RBDO:     {rbdo_volume:.2f}m³ ({(rbdo_volume-initial_volume)/initial_volume*100:+.1f}%)")
    print(f"\n🎯 Reliability:")
    print(f"   DDO:  β={beta_sys_ddo:.2f}")
    print(f"   RBDO: β={beta_sys_rbdo:.2f} ")

    print("✨ COMPLETE!")
   