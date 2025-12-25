import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

def generate_job_failure_data(n_samples: int = 1000, seed: int = 42) -> pd.DataFrame:
    """
    Generates synthetic SAP Job failure data (TBTCO/TBTCP style).
    """
    np.random.seed(seed)
    
    records = []
    job_classes = ['A', 'B', 'C']
    job_names = ['Z_FIN_POST', 'Z_SALES_EXTRACT', 'Z_INV_RECON', 'Z_HR_SYNC', 'Z_MRP_RUN']
    
    for i in range(n_samples):
        job_name = np.random.choice(job_names)
        job_class = np.random.choice(job_classes, p=[0.1, 0.3, 0.6])
        
        # Features
        duration_sec = np.random.gamma(shape=2, scale=300) # Avg 600s
        delay_sec = np.random.exponential(scale=100)
        step_count = np.random.randint(1, 15)
        concurrent_jobs = np.random.randint(0, 50)
        mem_usage_pct = np.random.uniform(10, 95)
        cpu_load_pct = np.random.uniform(5, 90)
        has_variant = np.random.choice([0, 1], p=[0.2, 0.8])
        hist_fail_rate = np.random.uniform(0, 0.15)
        
        # Non-linear risk formula
        # Risk increases with high concurrency, high memory, and high delay
        risk_score = (
            (concurrent_jobs / 50) * 1.5 +
            (mem_usage_pct / 100) * 2.0 +
            (delay_sec / 500) * 1.2 +
            (1 if job_class == 'A' else 0) * 0.5 +
            hist_fail_rate * 5.0
        )
        risk_score += np.random.normal(0, 0.2)
        
        # Determine class
        if risk_score > 3.5:
            status = 'Cancelled'
            risk_label = 'HIGH'
        elif risk_score > 2.2:
            status = 'Finished' # But risky
            risk_label = 'MEDIUM'
        else:
            status = 'Finished'
            risk_label = 'LOW'
            
        records.append({
            'JOBNAME': job_name,
            'JOBCOUNT': f'{i:08d}',
            'JOBCLASS': job_class,
            'DURATION_SEC': round(duration_sec, 1),
            'DELAY_SEC': round(delay_sec, 1),
            'STEP_COUNT': step_count,
            'CONCURRENT_JOBS': concurrent_jobs,
            'MEM_USAGE_PCT': round(mem_usage_pct, 1),
            'CPU_LOAD_PCT': round(cpu_load_pct, 1),
            'HAS_VARIANT': has_variant,
            'HIST_FAIL_RATE': round(hist_fail_rate, 3),
            'STATUS': status,
            'RISK_SCORE': round(risk_score, 2),
            'RISK_LABEL': risk_label
        })
        
    return pd.DataFrame(records)

def generate_transport_failure_data(n_samples: int = 1000, seed: int = 42) -> pd.DataFrame:
    """
    Generates synthetic SAP Transport failure data (E070/E071 style).
    """
    np.random.seed(seed)
    
    records = []
    users = ['DEV_ALAL', 'DEV_JDOE', 'DEV_BSMITH', 'DEV_KLEE']
    systems = ['DEV', 'QAS', 'PRD']
    
    for i in range(n_samples):
        user = np.random.choice(users)
        obj_count = np.random.randint(1, 500)
        table_obj_pct = np.random.uniform(0, 0.8)
        prog_obj_pct = 1.0 - table_obj_pct
        
        cross_sys_dep = np.random.randint(0, 10)
        author_success_rate = np.random.uniform(0.7, 0.99)
        target_sys_load = np.random.uniform(10, 90)
        network_latency = np.random.uniform(5, 200)
        
        # Risk formula
        risk_score = (
            (obj_count / 500) * 2.0 +
            table_obj_pct * 1.5 +
            cross_sys_dep * 0.5 +
            (1 - author_success_rate) * 4.0 +
            (target_sys_load / 100) * 1.0 +
            (network_latency / 200) * 0.8
        )
        risk_score += np.random.normal(0, 0.3)
        
        if risk_score > 4.0:
            risk_label = 'HIGH'
            result = 'Error'
        elif risk_score > 2.5:
            risk_label = 'MEDIUM'
            result = 'Warning'
        else:
            risk_label = 'LOW'
            result = 'Success'
            
        records.append({
            'TRKORR': f'SIDK9{i:05d}',
            'AS4USER': user,
            'OBJ_COUNT': obj_count,
            'TABLE_OBJ_PCT': round(table_obj_pct, 3),
            'PROG_OBJ_PCT': round(prog_obj_pct, 3),
            'CROSS_SYS_DEP': cross_sys_dep,
            'AUTHOR_SUCCESS_RATE': round(author_success_rate, 3),
            'TARGET_SYS_LOAD': round(target_sys_load, 1),
            'NETWORK_LATENCY': round(network_latency, 1),
            'RESULT': result,
            'RISK_SCORE': round(risk_score, 2),
            'RISK_LABEL': risk_label
        })
        
    return pd.DataFrame(records)

def generate_interface_failure_data(n_samples: int = 1000, seed: int = 42) -> pd.DataFrame:
    """
    Generates synthetic SAP Interface failure data (IDoc/RFC style).
    """
    np.random.seed(seed)
    
    records = []
    msg_types = ['ORDERS', 'INVOIC', 'MATMAS', 'DEBMAS']
    partners = ['CUST_A', 'VEND_B', 'SYS_X', 'EXT_Y']
    
    for i in range(n_samples):
        msg_type = np.random.choice(msg_types)
        partner = np.random.choice(partners)
        
        payload_size_kb = np.random.lognormal(mean=4, sigma=1)
        queue_depth = np.random.randint(0, 1000)
        partner_reliability = np.random.uniform(0.6, 0.99)
        retry_count = np.random.randint(0, 5)
        sys_load_idx = np.random.uniform(0.1, 0.9)
        dest_availability = np.random.uniform(0.5, 1.0)
        
        # Risk formula
        risk_score = (
            (payload_size_kb / 500) * 1.0 +
            (queue_depth / 1000) * 2.0 +
            (1 - partner_reliability) * 3.0 +
            retry_count * 0.8 +
            sys_load_idx * 1.5 +
            (1 - dest_availability) * 2.5
        )
        risk_score += np.random.normal(0, 0.25)
        
        if risk_score > 4.5:
            risk_label = 'HIGH'
            status = 'Error'
        elif risk_score > 2.8:
            risk_label = 'MEDIUM'
            status = 'Warning'
        else:
            risk_label = 'LOW'
            status = 'Success'
            
        records.append({
            'MESTYP': msg_type,
            'PARTNER': partner,
            'PAYLOAD_SIZE_KB': round(payload_size_kb, 1),
            'QUEUE_DEPTH': queue_depth,
            'PARTNER_RELIABILITY': round(partner_reliability, 3),
            'RETRY_COUNT': retry_count,
            'SYS_LOAD_IDX': round(sys_load_idx, 2),
            'DEST_AVAILABILITY': round(dest_availability, 3),
            'STATUS': status,
            'RISK_SCORE': round(risk_score, 2),
            'RISK_LABEL': risk_label
        })
        
    return pd.DataFrame(records)

def detect_drift(df1: pd.DataFrame, df2: pd.DataFrame, column: str) -> float:
    """
    Simple drift detection using mean difference percentage.
    """
    if column not in df1.columns or column not in df2.columns:
        return 0.0
    
    m1 = df1[column].mean()
    m2 = df2[column].mean()
    
    if m1 == 0: return 0.0
    return abs(m1 - m2) / m1
