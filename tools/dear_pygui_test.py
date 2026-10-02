import dearpygui.dearpygui as dpg

dpg.create_context()
dpg.create_viewport(title="RL Simulation Dashboard", width=1200, height=800)

# 1. Setup GPU Textures (Your background images)
with dpg.texture_registry():
    # width, height, channels, data = load_your_map_image()
    # dpg.add_static_texture(width, height, data, tag="env_background")
    pass

# 2. Main Layout (Left Control Panel + Center Simulation Grid)
with dpg.window(label="Main Window", tag="PrimaryWindow"):
    
    with dpg.group(horizontal=True):  # Holds sidebar and grid side-by-side
        
        # --- LEFT PANEL: CONTROLS & DATA ---
        with dpg.child_window(width=300, label="Simulation Controls"):
            dpg.add_text("Hyperparameters")
            dpg.add_slider_float(label="Learning Rate", default_value=0.001, max_value=0.01)
            dpg.add_slider_int(label="Batch Size", default_value=64, max_value=512)
            dpg.add_separator()
            dpg.add_text("Metrics")
            dpg.add_text("Avg Reward: 0.0", tag="reward_text")
            
        # --- CENTER PANEL: GRID OF SIMULATION TILES ---
        with dpg.child_window(label="Environment Viewports"):
            
            # Create a 2x2 grid using a DPG table structure
            with dpg.table(header_row=False, resizable=True, policy=dpg.mvTable_SizingStretchProp):
                dpg.add_table_column() # Column 1
                dpg.add_table_column() # Column 2
                
                # Row 1
                with dpg.table_row():
                    # Tile 1
                    with dpg.drawlist(width=400, height=300, tag="tile_1_canvas"):
                        dpg.draw_image("env_background", pmin=(0,0), pmax=(400,300))
                        dpg.draw_circle(center=(200, 150), radius=8, color=(255, 0, 0), tag="agent_1")
                    # Tile 2
                    with dpg.drawlist(width=400, height=300, tag="tile_2_canvas"):
                        dpg.draw_image("env_background", pmin=(0,0), pmax=(400,300))
                        dpg.draw_circle(center=(100, 100), radius=8, color=(0, 255, 0), tag="agent_2")
                
                # Row 2
                with dpg.table_row():
                    # Tile 3
                    with dpg.drawlist(width=400, height=300, tag="tile_3_canvas"):
                        dpg.draw_image("env_background", pmin=(0,0), pmax=(400,300))
                    # Tile 4
                    with dpg.drawlist(width=400, height=300, tag="tile_4_canvas"):
                        dpg.draw_image("env_background", pmin=(0,0), pmax=(400,300))

dpg.set_primary_window("PrimaryWindow", True)
dpg.setup_dearpygui()
dpg.show_viewport()

# 3. Main RL Simulation Loop
while dpg.is_dearpygui_running():
    # A. Step your RL environments here
    # new_coords = step_simulations()
    
    # B. Update positions on the GPU instantly using tags
    # dpg.configure_item("agent_1", center=(new_coords[0]))
    # dpg.configure_item("agent_2", center=(new_coords[1]))
    
    # C. Update text readouts
    # dpg.set_value("reward_text", f"Avg Reward: {current_reward}")
    
    dpg.render_dearpygui_frame()

dpg.destroy_context()
