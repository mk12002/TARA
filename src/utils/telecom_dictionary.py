"""
Telecom Acronym Dictionary
=============================
3000+ telecom acronyms for query enrichment and NER.
"""

# Core set — expand as needed during development
TELECOM_ACRONYMS = {
    # Radio Access
    "NR": "New Radio",
    "LTE": "Long Term Evolution",
    "RAN": "Radio Access Network",
    "RRC": "Radio Resource Control",
    "PDCCH": "Physical Downlink Control Channel",
    "PDSCH": "Physical Downlink Shared Channel",
    "PUCCH": "Physical Uplink Control Channel",
    "PUSCH": "Physical Uplink Shared Channel",
    "PRACH": "Physical Random Access Channel",
    "PRB": "Physical Resource Block",
    "UE": "User Equipment",
    "GNB": "gNodeB (5G Base Station)",
    "ENB": "eNodeB (LTE Base Station)",
    "SSB": "Synchronization Signal Block",
    "CSI": "Channel State Information",
    "CQI": "Channel Quality Indicator",
    "MCS": "Modulation and Coding Scheme",
    "HARQ": "Hybrid Automatic Repeat Request",
    "DRX": "Discontinuous Reception",
    "BWP": "Bandwidth Part",
    "SCS": "Subcarrier Spacing",
    "CORESET": "Control Resource Set",

    # KPIs & Measurements
    "RSRP": "Reference Signal Received Power",
    "RSRQ": "Reference Signal Received Quality",
    "SINR": "Signal to Interference plus Noise Ratio",
    "BLER": "Block Error Rate",
    "MTTR": "Mean Time To Repair",
    "MTTF": "Mean Time To Failure",
    "KPI": "Key Performance Indicator",
    "QoS": "Quality of Service",
    "QoE": "Quality of Experience",
    "VSWR": "Voltage Standing Wave Ratio",

    # Architecture
    "O-RAN": "Open Radio Access Network",
    "ORAN": "Open Radio Access Network",
    "CU": "Central Unit",
    "DU": "Distributed Unit",
    "RU": "Radio Unit",
    "RIC": "RAN Intelligent Controller",
    "SMO": "Service Management and Orchestration",
    "AMF": "Access and Mobility Management Function",
    "SMF": "Session Management Function",
    "UPF": "User Plane Function",
    "NRF": "Network Repository Function",
    "PCF": "Policy Control Function",

    # Protocols
    "PDCP": "Packet Data Convergence Protocol",
    "RLC": "Radio Link Control",
    "MAC": "Medium Access Control",
    "SDAP": "Service Data Adaptation Protocol",
    "GTP": "GPRS Tunnelling Protocol",
    "SCTP": "Stream Control Transmission Protocol",
    "NGAP": "NG Application Protocol",
    "XnAP": "Xn Application Protocol",
    "F1AP": "F1 Application Protocol",
    "E1AP": "E1 Application Protocol",

    # Procedures
    "HO": "Handover",
    "RCA": "Root Cause Analysis",
    "ANR": "Automatic Neighbor Relations",
    "MLB": "Mobility Load Balancing",
    "MRO": "Mobility Robustness Optimization",
    "CCO": "Coverage and Capacity Optimization",
    "SON": "Self-Organizing Network",
    "MDT": "Minimization of Drive Tests",

    # Technologies
    "MIMO": "Multiple Input Multiple Output",
    "CA": "Carrier Aggregation",
    "DC": "Dual Connectivity",
    "DSS": "Dynamic Spectrum Sharing",
    "NSA": "Non-Standalone",
    "SA": "Standalone",
    "MEC": "Multi-access Edge Computing",
    "NFV": "Network Functions Virtualization",
    "SDN": "Software Defined Networking",

    # Spectrum
    "FR1": "Frequency Range 1 (sub-6 GHz)",
    "FR2": "Frequency Range 2 (mmWave)",
    "TDD": "Time Division Duplex",
    "FDD": "Frequency Division Duplex",
    "SUL": "Supplementary Uplink",

    # Slicing & Services
    "eMBB": "Enhanced Mobile Broadband",
    "URLLC": "Ultra-Reliable Low-Latency Communications",
    "mMTC": "Massive Machine Type Communications",
    "NSI": "Network Slice Instance",
    "NSSI": "Network Slice Subnet Instance",

    # Standards
    "3GPP": "3rd Generation Partnership Project",
    "ETSI": "European Telecommunications Standards Institute",
    "ITU": "International Telecommunication Union",
    "TS": "Technical Specification",
    "TR": "Technical Report",
}
