from MODGenX.vis_3d_models import plot_3d_model, plot_cross_section
import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser(description='Visualize MODFLOW model')
    parser.add_argument('--interactive', action='store_true', help='Enable interactive visualization (requires X server)')
    parser.add_argument('--no-screenshot', dest='save_screenshot', action='store_false', help='Disable saving screenshots')
    parser.add_argument('--no-html', dest='save_html', action='store_false', help='Disable saving HTML visualizations')
    parser.add_argument('--xvfb', action='store_true', help='Use xvfb-run to start a virtual X server')
    parser.add_argument('--ve', '--vertical-exaggeration', type=float, default=5.0, 
                        help='Vertical exaggeration factor (default: 5.0)')
    args = parser.parse_args()
    
    # If xvfb flag is set, restart the script using xvfb-run
    if args.xvfb and not os.environ.get('XVFB_RUNNING'):
        os.environ['XVFB_RUNNING'] = '1'
        cmd = f"xvfb-run -a {sys.executable} {' '.join(sys.argv)}"
        print(f"Restarting with xvfb-run: {cmd}")
        os.system(cmd)
        return 0

    name = "04112500"
    username = "vahidr32"
    vpuid = "0405"
    level = "huc12"
    model = "MODFLOW_250m"

    base_path = f"/data/SWATGenXApp/Users/{username}/SWATplus_by_VPUID/{vpuid}/{level}/{name}/{model}/"

    # Check if base path exists
    if not os.path.exists(base_path):
        print(f"Error: Base path does not exist: {base_path}")
        return 1

    try:
        # Call the 3D visualization function with all parameters
        plot_3d_model(
            base_path, 
            model, 
            save_screenshot=args.save_screenshot, 
            interactive=args.interactive,
            vertical_exaggeration=args.ve,
            save_html=args.save_html
        )
        print("Visualization complete")
        
        # Print guidance on viewing the visualizations
        output_dir = os.path.join(base_path, "visualization")
        if os.path.exists(output_dir):
            html_files = [f for f in os.listdir(output_dir) if f.endswith('.html')]
            if html_files:
                print("\nTo view the interactive 3D visualizations:")
                print(f"1. Navigate to: {output_dir}")
                print("2. Open one of the following HTML files in a web browser:")
                for html_file in html_files:
                    print(f"   - {html_file}")
                print("\nNote: For the best experience, use Chrome or Firefox.")
        
        return 0
    except Exception as e:
        print(f"Error during visualization: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
