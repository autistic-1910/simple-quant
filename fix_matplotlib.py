import matplotlib
import sys
import os

def fix_matplotlib_backend():
    """Fix matplotlib backend issues"""
    
    print("=== Matplotlib Backend Fix ===\n")
    
    # Check current backend
    current_backend = matplotlib.get_backend()
    print(f"Current backend: {current_backend}")
    
    # Try different backends
    backends_to_try = ['TkAgg', 'Qt5Agg', 'Agg']
    
    for backend in backends_to_try:
        try:
            matplotlib.use(backend, force=True)
            import matplotlib.pyplot as plt
            
            # Test if backend works
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot([1, 2, 3], [1, 4, 2])
            ax.set_title(f"Test plot with {backend} backend")
            
            if backend != 'Agg':  # Agg doesn't show plots
                plt.show()
            else:
                plt.savefig('test_plot.png')
                print("✓ Plot saved as test_plot.png")
            
            plt.close()
            print(f"✓ {backend} backend works!")
            return backend
            
        except Exception as e:
            print(f"✗ {backend} backend failed: {e}")
            continue
    
    print("⚠ No working GUI backend found. Using Agg (save plots only)")
    matplotlib.use('Agg', force=True)
    return 'Agg'

def create_matplotlib_config():
    """Create matplotlib configuration file"""
    
    config_dir = matplotlib.get_configdir()
    config_file = os.path.join(config_dir, 'matplotlibrc')
    
    print(f"\nCreating matplotlib config at: {config_file}")
    
    config_content = """
# Matplotlib configuration for Quantitative Finance Tool
backend: TkAgg
figure.figsize: 10, 6
figure.dpi: 100
savefig.dpi: 150
font.size: 10
axes.titlesize: 12
axes.labelsize: 10
xtick.labelsize: 9
ytick.labelsize: 9
legend.fontsize: 9
"""
    
    try:
        os.makedirs(config_dir, exist_ok=True)
        with open(config_file, 'w') as f:
            f.write(config_content.strip())
        print("✓ Matplotlib configuration created")
    except Exception as e:
        print(f"✗ Failed to create config: {e}")

if __name__ == "__main__":
    working_backend = fix_matplotlib_backend()
    create_matplotlib_config()
    
    print(f"\n=== Summary ===")
    print(f"Working backend: {working_backend}")
    print("\nTo use this backend in your scripts, add this at the top:")
    print(f"import matplotlib")
    print(f"matplotlib.use('{working_backend}')")
    print("import matplotlib.pyplot as plt")