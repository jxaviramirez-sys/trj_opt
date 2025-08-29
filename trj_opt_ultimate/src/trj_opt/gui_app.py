# Reconstructed by merge of 1 variants for trj_opt/gui_app.py
# Strategy: union of imports, union of classes/functions with longest/most-params preference.

import os
import pandas as pd
import streamlit as st
from trj_opt.calibration import CalibrationSettings, load_settings, calibrate_csv, load_measured_points, compute_anchor_multipliers, apply_fudges_to_dataframe
from trj_opt.pareto import plot_pareto_with_overlay
import os
import pandas as pd
import streamlit as st
from trj_opt.calibration import CalibrationSettings, load_settings, calibrate_csv, load_measured_points, compute_anchor_multipliers, apply_fudges_to_dataframe
from trj_opt.pareto import plot_pareto_with_overlay
st.set_page_config(page_title='TRJ Opt – Calibration', layout='centered')
st.title('TRJ Optimization – Quick Calibration GUI')
st.markdown('\nUse this mini-GUI to apply calibration knobs and overlay measured data on top of your analyzed DOE grid.\n- Load your analyzed CSV (output from `trj_opt analyze` or demo).\n- Tweak fudge factors and/or provide an anchor measured ρc.\n- Optionally upload a CSV of measured points to overlay on the Pareto.\n- Download the calibrated CSV and Pareto image.\n')
uploaded = st.file_uploader('Analyzed CSV', type=['csv'])
measured = st.file_uploader('Measured points CSV (optional)', type=['csv'])
col1, col2, col3 = st.columns(3)
with col1:
    fudge_rho_c = st.number_input('Fudge ρc (×)', min_value=0.0, value=1.0, step=0.01, format='%.3f')
with col2:
    fudge_J0 = st.number_input('Fudge J0_interface (×)', min_value=0.0, value=1.0, step=0.01, format='%.3f')
with col3:
    anchor_rhoc = st.text_input('Anchor ρc (Ohm·cm²)', value='')
if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.write('Preview:', df.head())
    settings = CalibrationSettings(fudge_rho_c=float(fudge_rho_c), fudge_J0_interface=float(fudge_J0), anchor_rho_c_Ohm_cm2=float(anchor_rhoc) if anchor_rhoc.strip() else None)
    if st.button('Apply calibration'):
        settings2 = compute_anchor_multipliers(df, settings)
        out_df = apply_fudges_to_dataframe(df, settings2)
        st.session_state['calib_df'] = out_df
        st.success('Calibration applied.')
    if 'calib_df' in st.session_state:
        out_df = st.session_state['calib_df']
        st.download_button('Download calibrated CSV', data=out_df.to_csv(index=False), file_name='analyzed_calibrated.csv', mime='text/csv')
        m_df = None
        if measured is not None:
            try:
                m_df = pd.read_csv(measured)
            except Exception as e:
                st.warning(f'Could not read measured points CSV: {e}')
        try:
            xcol = 'rho_c_mOhm_cm2'
            ycol = 'DeltaVoc_interface_mV'
            plot_pareto_with_overlay(out_df, xcol, ycol, 'pareto_overlay.png', title='Contact vs. ΔVoc penalty', measured_df=m_df)
            st.image('pareto_overlay.png')
            with open('pareto_overlay.png', 'rb') as f:
                st.download_button('Download Pareto (PNG)', data=f, file_name='pareto_overlay.png', mime='image/png')
        except Exception as e:
            st.warning(f'Could not plot Pareto: {e}')
else:
    st.info('Upload an analyzed CSV to begin.')
