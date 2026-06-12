import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

def setup_fuzzy_system():
    print("Setting up Phase 2: Fuzzy Logic Severity System...")
    
    # 1. Define the inputs (Antecedents) and output (Consequent)
    # In a dashcam, a pothole covering 25% of the image is massive.
    area = ctrl.Antecedent(np.arange(0, 26, 1), 'area')
    depth = ctrl.Antecedent(np.arange(0, 11, 1), 'depth')
    
    # The output is the Severity Score (0 to 100)
    severity = ctrl.Consequent(np.arange(0, 101, 1), 'severity')

    # 2. Create the Fuzzy Sets (Low, Medium, High)
    # Custom, realistic sizes for Area in a dashcam view
    area['small'] = fuzz.trimf(area.universe, [0, 0, 6])
    area['medium'] = fuzz.trimf(area.universe, [4, 10, 16])
    area['large'] = fuzz.trimf(area.universe, [12, 25, 25])
    
    # Custom depths
    depth['shallow'] = fuzz.trimf(depth.universe, [0, 0, 4])
    depth['moderate'] = fuzz.trimf(depth.universe, [2, 5, 8])
    depth['deep'] = fuzz.trimf(depth.universe, [6, 10, 10])
    
    # Custom fuzzy sets for Severity Output
    severity['low'] = fuzz.trimf(severity.universe, [0, 0, 45])
    severity['medium'] = fuzz.trimf(severity.universe, [30, 50, 70])
    severity['high'] = fuzz.trimf(severity.universe, [55, 100, 100])

    # 3. Create the Rules (UPDATED: Giving Depth much more importance)
    
    # Rule 1: It must be small AND shallow to be low priority.
    rule1 = ctrl.Rule(area['small'] & depth['shallow'], severity['low'])
    
    # Rule 2: If it is small but moderate depth, OR medium but shallow, it becomes medium priority.
    rule2 = ctrl.Rule((area['small'] & depth['moderate']) | (area['medium'] & depth['shallow']), severity['medium'])
    
    # Rule 3: If it is large, OR deep, OR (medium area AND moderate depth), it is high priority!
    rule3 = ctrl.Rule(area['large'] | depth['deep'] | (area['medium'] & depth['moderate']), severity['high'])

    # 4. Build the Control System
    severity_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
    severity_sim = ctrl.ControlSystemSimulation(severity_ctrl)
    
    return severity_sim

def calculate_severity(area_val, depth_val, severity_sim):
    # Pass the inputs into the system (cap area at 25 so we don't break the scale)
    severity_sim.input['area'] = min(area_val, 25)
    severity_sim.input['depth'] = depth_val
    
    # Compute the result
    severity_sim.compute()
    
    score = severity_sim.output['severity']
    
    # Classify the score 
    if score < 40:
        category = "Low Priority"
    elif score < 70:
        category = "Medium Priority"
    else:
        category = "High Priority"
        
    return score, category

if __name__ == '__main__':
    # Make sure to install the library first: pip install scikit-fuzzy
    system = setup_fuzzy_system()
    
    # Let's test it with a fake pothole
    test_area = 5   # Small area
    test_depth = 6  # Moderate/Deep depth
    
    print(f"\nTesting with Area: {test_area}, Depth: {test_depth}")
    score, category = calculate_severity(test_area, test_depth, system)
    
    print(f"Severity Score: {score:.2f}/100")
    print(f"Classification: {category}")