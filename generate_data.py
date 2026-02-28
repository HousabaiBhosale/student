import pandas as pd
import numpy as np
import os

def generate_synthetic_data(num_records=1200, output_dir='.'):
    """
    Generates synthetic student performance data.
    Features: Attendance(%), StudyHours, AssignScore, PrevGPA, Participation, NetUsage, Sleep, FamilySupport, ExtraCurr.
    Targets: FinalScore, Pass, Performance, Risk.
    """
    np.random.seed(42)  # For reproducibility

    # Generate Features
    attendance = np.round(np.random.uniform(50, 100, num_records)).astype(int)
    study_hours = np.round(np.random.uniform(1, 10, num_records), 1)
    assign_score = np.round(np.random.uniform(40, 100, num_records)).astype(int)
    prev_gpa = np.round(np.random.uniform(4, 10, num_records), 1)
    participation = np.random.randint(0, 11, num_records)
    net_usage = np.round(np.random.uniform(1, 8, num_records), 1)
    sleep = np.round(np.random.uniform(4, 10, num_records), 1)
    fam_support = np.random.randint(0, 11, num_records)
    extra_curr = np.random.randint(0, 11, num_records)

    # Calculate base score based on features (similar logic to JS frontend)
    # Weights are approximations to create a realistic distribution
    raw_score = (attendance * 0.14) + (study_hours * 3) + (assign_score * 0.18) + \
                (prev_gpa * 2.5) + (participation * 0.8) - (net_usage * 0.5) + \
                (sleep * 0.5) + (fam_support * 0.4) + (extra_curr * 0.2)
    
    # Add some random noise
    noise = np.random.normal(0, 5, num_records)
    final_score = np.round(np.clip(raw_score + noise, 0, 100)).astype(int)

    # Derive Targets
    pass_fail = ['PASS' if score >= 50 else 'FAIL' for score in final_score]
    
    performance = []
    for score in final_score:
        if score >= 85:
            performance.append('Excellent')
        elif score >= 70:
            performance.append('Good')
        elif score >= 50:
            performance.append('Average')
        else:
            performance.append('Poor')

    risk = []
    for i in range(num_records):
        att = attendance[i]
        score = final_score[i]
        if att > 80 and score > 70:
            risk.append('Low')
        elif att > 60 and score > 50:
            risk.append('Medium')
        else:
            risk.append('High')

    # Create DataFrame
    data = {
        'Attendance': attendance,
        'StudyHours': study_hours,
        'AssignScore': assign_score,
        'PrevGPA': prev_gpa,
        'Participation': participation,
        'NetUsage': net_usage,
        'Sleep': sleep,
        'FamilySupport': fam_support,
        'ExtraCurr': extra_curr,
        'FinalScore': final_score,
        'Pass': pass_fail,
        'Performance': performance,
        'Risk': risk
    }

    df = pd.DataFrame(data)
    
    # Ensure directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to CSV
    filepath = os.path.join(output_dir, 'student_data.csv')
    df.to_csv(filepath, index=False)
    print(f"Generated {num_records} records of synthetic data at {filepath}")

if __name__ == '__main__':
    generate_synthetic_data(1200, 'data')
