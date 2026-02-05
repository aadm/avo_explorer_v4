# avo_explorer_v4
# -----------------
# aadm 2019, 2023, 2024
#
# streamlit port of the clunky v2 jupyter notebook+docker
# found at https://github.com/aadm/avo_explorer
# 
# to run locally: 
# $ streamlit run avo_explorer_v3.py
#
# to run webapp:
#
# https://xxx.streamlit.app/
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import numpy as np
import altair as alt
import pandas as pd
import streamlit as st


def pr(vp, vs):
    '''
    Calculate Poisson's ratio. (aadm 2017)
    '''
    return (vp**2-2*vs**2) / (2*(vp**2-vs**2))


def hilterman(vp1, vs1, rho1, vp2, vs2, rho2, theta,
         angcontrib=False):
    '''
    Calculate P-wave reflectivity with Verm & Hilterman's equation. (aadm 2024)

    Hilterman (2001), Seismic Amplitude Interpretation,
    SEG Distinguished Instructor Short Course n.4 (p. 3-35)
    '''
    a = np.deg2rad(theta)
    pr1 = pr(vp1, vs1)
    pr2 = pr(vp2, vs2)
    dvp = vp2 - vp1
    drho = rho2 - rho1
    dpr = pr2 - pr1

    # calculate average properties
    avp = np.mean([vp1, vp2], axis=0)
    avs = np.mean([vs1, vs2], axis=0)
    arho = np.mean([rho1, rho2], axis=0)
    apr = np.mean([pr1, pr2], axis=0)
 
    R0 = 0.5 * (dvp/avp + drho/arho) # normal incidence reflection coefficient

    C = 4 * avs**2/avp**2 * np.sin(a)**2
    term1 = R0 * (1 - 4 * avs**2/avp**2 * np.sin(a)**2)
    term2 = dpr/(1 - apr)**2 * np.sin(a)**2
    term3 = 0.5 * dvp/avp * (np.tan(a)**2 - 4 * avs**2/avp**2 * np.sin(a)**2)
    R = term1 + term2 + term3

    if angcontrib:
        return R, term1, term2, term3
    else:
        return R


def shuey(vp1, vs1, rho1, vp2, vs2, rho2, theta,
           approx=True, terms=False, angcontrib=False):
    '''
    Calculate P-wave reflectivity with Shuey's equation. (aadm 2024)

    Mavko et al. (2009), The Rock Physics Handbook, 2nd ed.
      Cambridge University Press (p.102)
    Castagna (1993), AVO Analysis - Tutorial and Review in
      "Offset-Dependent Reflectivity: Theory and Practice of AVO Analysis",
      ed. J. P. Castagna and M. Backus, SEG Investigations in Geophysics n.8 (p.9)
    '''
    a = np.deg2rad(theta)
    pr1 = pr(vp1, vs1)
    pr2 = pr(vp2, vs2)
    dvp = vp2 - vp1
    drho = rho2 - rho1
    dpr = pr2 - pr1

    # calculate average properties
    avp = np.mean([vp1, vp2], axis=0)
    arho = np.mean([rho1, rho2], axis=0)
    apr = np.mean([pr1, pr2], axis=0)

    R0 = 0.5*(dvp/avp + drho/arho) # normal incidence reflection coefficient

    B = (dvp/avp) / (dvp/avp + drho/arho)
    A0 = B - 2*(1 + B) * (1 - 2*apr)/(1 - apr) 
    G = A0*R0 + dpr / (1 - apr)**2 # gradient: intermediate angles

    F = 0.5 * dvp/avp # F: dominant approaching critical angles

    # if angles _and_ velocities are array reshape
    if isinstance(theta, np.ndarray) & isinstance(vp1, np.ndarray):
        R0 = R0.reshape(-1, 1)
        G = G.reshape(-1, 1)
        F = F.reshape(-1, 1)

    R = R0 + G*np.sin(a)**2 + F*(np.tan(a)**2 - np.sin(a)**2)

    if approx:
        R = R0 + G*np.sin(a)**2
    else:
        R = R0 + G*np.sin(a)**2 + F*(np.tan(a)**2 - np.sin(a)**2)

    if terms:
        if angcontrib:
            return R, R0, G*np.sin(a)**2, F*(np.tan(a)**2 - np.sin(a)**2)
        else:
            return R, R0, G
    else:
        return R    


def get_avo_classes(hybrid_class_4=True):
    '''
    Returns reference AVO classes definition from 
    from Hilterman, 2001, Seismic Amplitude Interpretation,
    SEG-EAGE Distinguished Instructor Short Course.
    Class 4 from Castagna, J. P., and H. W. Swan, 1997,
    Principles of AVO crossplotting, The Leading Edge.
    '''
    tmp_shale = np.array([[3094, 1515, 2.40, 0],
                          [2643, 1167, 2.29, 0],
                          [2192, 818, 2.16, 0],
                          [3240, 1620, 2.34, 0]])
    tmp_sandg = np.array([[4050, 2526, 2.21, .2],
                          [2781, 1665, 2.08, .25],
                          [1542, 901, 1.88, .33],
                          [1650, 1090, 2.07, .163]])
    tmp_sandb = np.array([[4115, 2453, 2.32, .2],
                          [3048, 1595, 2.23, .25],
                          [2134, 860, 2.11, .33],
                          [2590, 1060, 2.21, .163]])
    avocl = ['CLASS1', 'CLASS2', 'CLASS3', 'CLASS4']
    logs = ['VP', 'VS', 'RHO', 'PHI']
    shale = pd.DataFrame(tmp_shale, columns=logs, index=avocl)
    sandg = pd.DataFrame(tmp_sandg, columns=logs, index=avocl)
    sandb = pd.DataFrame(tmp_sandb, columns=logs, index=avocl)
    if hybrid_class_4:
        sandb.loc['CLASS4'] = sandb.loc['CLASS3']
        sandg.loc['CLASS4'] = sandg.loc['CLASS3']
    return shale, sandb, sandg


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# initialize app

st.set_page_config(page_title='AVO Explorer v4', layout="centered")

with st.sidebar:
    st.title(':grey[AVO Explorer v4]')
    st.write(
        '''Porting of my old
        [AVO Explorer notebook](https://github.com/aadm/avo_explorer).
        (aadm 2019, 2023, 2024, 2025)''')

# st.title(':grey[AVO Explorer v4]')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# get elastic properties for default avo classes
sh, ssb, ssg = get_avo_classes()
avo_class_options = sh.index.to_list()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# widget: input elastic properties for shale and sand 

opt_vp = dict(min_value=1500., max_value=6000., step=10., format='%.0f')
opt_vs = dict(min_value=700., max_value=4000., step=10., format='%.0f')
opt_rho = dict(min_value=1.5, max_value=3.5, step=0.05, format='%.2f')
ep_ss = np.full(3, np.nan)
ep_sh = np.full(3, np.nan)

st.write('**UPPER LAYER**')
elprops_above = st.columns(3, gap='large')
with elprops_above[0]:
    ep_sh[0] = st.number_input('Vp', value=2200., **opt_vp)
with elprops_above[1]:
    ep_sh[1] = st.number_input('Vs', value=820.,  **opt_vs)
with elprops_above[2]:
    ep_sh[2] = st.number_input('rho', value=2.2, **opt_rho)
st.write('**LOWER LAYER**')
elprops_below = st.columns(3, gap='large')
with elprops_below[0]:
    ep_ss[0] = st.number_input('Vp', value=1550., **opt_vp)
with elprops_below[1]:
    ep_ss[1] = st.number_input('Vs', value=900.,  **opt_vs)
with elprops_below[2]:
    ep_ss[2] = st.number_input('rho', value=1.9, **opt_rho)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# widget: select angle range

aa = np.arange(0,91)
angle_range = st.select_slider(
    'Angle range', options=aa, value=[0.0, 30.0])
angles = np.arange(angle_range[0], angle_range[1])
if angles.size == 0:
    angles = np.array([angle_range[0]])

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# widget: misc options

cols0 = st.columns(2, gap='large', vertical_alignment='center')
with cols0[0]:
    refl_eq = st.segmented_control('Reflectivity equation', ['Shuey', 'Hilterman'], default='Shuey')
with cols0[1]:
    contrib_toggle = st.toggle('Plot angle contribution')

st.divider()

# cols1 = st.columns(3, gap='large')
# with cols1[0]:
#     avo_toggle = st.toggle('Plot AVO reference')
# with cols1[1]:
#     selected_class = st.selectbox('AVO Class', avo_class_options, index=2)
# with cols1[2]:
#     avoref = st.segmented_control('AVO model', ['Brine Sand', 'Gas Sand'], default='Gas Sand')


# st.divider()


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# make plot

def plot_rc(
    data,
    ang,
    refl_eq,
    reference=None,
    contrib=False,
    reference_label=None,
    xlim=None,
    ylim=None,
    reference_color=None,
):
    '''Return Altair chart of reflection coefficient vs angle.'''
    angles = np.asarray(ang).reshape(-1)
    chart_segments = []
    component_styles = {}

    def add_segment(values, label, color, style='solid', width=3):
        vals = np.asarray(values).reshape(-1)
        segment = pd.DataFrame({
            'Angle': angles,
            'Amplitude': vals,
            'Component': label,
            'Style': style
        })
        chart_segments.append(segment)
        if label not in component_styles:
            component_styles[label] = {'color': color, 'width': width}

    add_segment(data[0], f'[{refl_eq}] RC', '#000000', width=4)

    if contrib:
        if refl_eq == 'Shuey':
            add_segment(data[1] + data[2], f'[{refl_eq}] Near', '#8c564b')
            add_segment(data[3], f'[{refl_eq}] Far', '#ff7f0e')
        else:
            add_segment(data[1], f'[{refl_eq}] Near', '#8c564b')
            add_segment(data[2], f'[{refl_eq}] Mid', '#9467bd')
            add_segment(data[3], f'[{refl_eq}] Far', '#ff7f0e')

    if reference is not None:
        label = f'{reference_label} Reference' if reference_label else 'Reference'
        ref_color = reference_color or '#1f77b4'
        add_segment(reference, label, ref_color, style='dashed', width=3)

    chart_data = pd.concat(chart_segments, ignore_index=True)
    style_domain = chart_data['Style'].unique().tolist()
    dash_map = {'solid': [1, 0], 'dashed': [6, 3]}
    style_range = [dash_map.get(style, [1]) for style in style_domain]

    component_order = list(component_styles.keys())
    component_colors = [component_styles[label]['color'] for label in component_order]
    component_widths = [component_styles[label]['width'] for label in component_order]

    x_scale = alt.Scale(domain=xlim) if xlim else alt.Undefined
    y_scale = alt.Scale(domain=ylim) if ylim else alt.Undefined

    chart = (
        alt.Chart(chart_data)
        .mark_line(size=3)
        .encode(
            x=alt.X('Angle:Q', title='Angle of incidence', scale=x_scale),
            y=alt.Y('Amplitude:Q', title='Amplitude', scale=y_scale),
            color=alt.Color(
                'Component:N',
                scale=alt.Scale(domain=component_order, range=component_colors),
                legend=alt.Legend(title='Component', orient='bottom', direction='horizontal'),
            ),
            strokeDash=alt.StrokeDash('Style:N', scale=alt.Scale(domain=style_domain, range=style_range), legend=None),
            strokeWidth=alt.StrokeWidth(
                'Component:N',
                scale=alt.Scale(domain=component_order, range=component_widths),
                legend=None,
            ),
        )
        .properties(height=350)
    )

    return chart


cols1 = st.columns([0.25, 0.8], gap='large')
with cols1[0]:
    avo_toggle = st.toggle('AVO reference')
    selected_class = st.selectbox('AVO Class', avo_class_options, index=2)
    avoref = st.radio('AVO model', ['Brine Sand', 'Gas Sand'], index=0)

    logs = ['VP', 'VS', 'RHO']
    avorefsh = sh.loc[selected_class, logs]
    if avoref == 'Brine Sand':
        avorefss = ssb.loc[selected_class, logs]
    else:
        avorefss = ssg.loc[selected_class, logs]
    if refl_eq == 'Shuey':
        rc_ref = shuey(*avorefsh, *avorefss, angles, approx=False)
    else:
        rc_ref = hilterman(*avorefsh, *avorefss, angles)

    reference_color = '#1f77b4' if avoref == 'Brine Sand' else '#d62728'


    # calculate reflectivity from user input
    if refl_eq == 'Shuey':
        data = shuey(*ep_sh, *ep_ss, angles, approx=False, terms=True, angcontrib=True)
    else:
        data = hilterman(*ep_sh, *ep_ss, angles, angcontrib=True)

    # axis limit state
    rc_values = np.asarray(data[0]).reshape(-1)
    angle_values = np.asarray(angles).reshape(-1)
    defaults = {
        'axis_x_min': float(angle_values.min()) if angle_values.size else 0.0,
        'axis_x_max': float(angle_values.max()) if angle_values.size else 0.0,
        'axis_y_min': float(rc_values.min()) if rc_values.size else -0.5,
        'axis_y_max': float(rc_values.max()) if rc_values.size else 0.5,
        'lock_x_axis': False,
        'lock_y_axis': False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    def _get_axis_limits(lock_key, min_key, max_key):
        if not st.session_state.get(lock_key):
            return None
        vmin = float(st.session_state.get(min_key))
        vmax = float(st.session_state.get(max_key))
        if np.isclose(vmin, vmax):
            return None
        low, high = sorted([vmin, vmax])
        return [low, high]

    x_limits = _get_axis_limits('lock_x_axis', 'axis_x_min', 'axis_x_max')
    y_limits = _get_axis_limits('lock_y_axis', 'axis_y_min', 'axis_y_max')

with cols1[1]:
    if avo_toggle:
        avo_chart = plot_rc(
            data,
            angles,
            refl_eq,
            reference=rc_ref,
            contrib=contrib_toggle,
            reference_label=selected_class,
            xlim=x_limits,
            ylim=y_limits,
            reference_color=reference_color,
        )
    else:
        avo_chart = plot_rc(
            data,
            angles,
            refl_eq,
            contrib=contrib_toggle,
            xlim=x_limits,
            ylim=y_limits,
        )
    st.altair_chart(avo_chart, width=500, height=500)




st.divider()
axis_cols = st.columns(2, gap='large')
with axis_cols[0]:
    st.write('**X axis**')
    st.checkbox('Lock x-axis', key='lock_x_axis')
    st.number_input('Min angle', key='axis_x_min', value=st.session_state['axis_x_min'], step=1.0)
    st.number_input('Max angle', key='axis_x_max', value=st.session_state['axis_x_max'], step=1.0)
with axis_cols[1]:
    st.write('**Y axis**')
    st.checkbox('Lock y-axis', key='lock_y_axis')
    st.number_input('Min amplitude', key='axis_y_min', value=st.session_state['axis_y_min'], step=0.05, format='%.3f')
    st.number_input('Max amplitude', key='axis_y_max', value=st.session_state['axis_y_max'], step=0.05, format='%.3f')

