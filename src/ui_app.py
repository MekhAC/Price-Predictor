import io
import os
import re
import tempfile
from contextlib import redirect_stdout
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from batch_predict import predict_batch
from data_preprocessing import is_native_lightgbm_preprocessor, prepare_model_input
from demand_adjuster import DemandAdjuster

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODELS_DIR = os.path.join(BASE_DIR, 'models')


def _read_any(path: str) -> pd.DataFrame:
    if path.lower().endswith(('.xlsx', '.xls')):
        return pd.read_excel(path)
    return pd.read_csv(path)


@st.cache_resource
def _load_artifacts():
    meta = joblib.load(os.path.join(MODELS_DIR, 'preprocessor.joblib'))
    if not is_native_lightgbm_preprocessor(meta):
        raise RuntimeError(
            'preprocessor.joblib is from the old OHE pipeline. Re-run src/train_model.py once to rebuild artifacts.'
        )

    model = joblib.load(os.path.join(MODELS_DIR, 'price_model.joblib'))
    adjuster = DemandAdjuster()
    return meta, model, adjuster


def _resolve_reg_state(reg_state: str, registration: str) -> str:
    if reg_state and reg_state.strip():
        return reg_state.strip().upper()[:2]

    registration = (registration or '').strip().upper()
    if len(registration) >= 2 and registration[:2].isalpha():
        return registration[:2]

    return 'UNK'


def _single_predict(
    city: str,
    make: str,
    model_name: str,
    variant: str,
    transmission: str,
    fuel: str,
    bodytype: str,
    year: int,
    ownership: float,
    kms_driven: float,
    registration: str,
    reg_state: str,
):
    meta, model, adjuster = _load_artifacts()

    resolved_state = _resolve_reg_state(reg_state, registration)
    current_year = datetime.now().year
    car_age = max(current_year - int(year), 0)
    kms_per_year = float(kms_driven) / car_age if car_age > 0 else float(kms_driven)

    row = pd.DataFrame([
        {
            'City': city,
            'Make': make,
            'Model': model_name,
            'Variant': variant,
            'Transmission': transmission,
            'Fuel': fuel,
            'BodyType': bodytype,
            'Year': int(year),
            'KMs Driven': float(kms_driven),
            'Ownership': float(ownership),
            'Reg State': resolved_state,
            'Car Age': float(car_age),
            'KMs/Year': float(kms_per_year),
        }
    ])

    X_proc = prepare_model_input(row, meta)
    base_price = float(model.predict(X_proc)[0])

    make_value = str(row.at[0, 'Make'])
    model_value = str(row.at[0, 'Model'])
    variant_value = str(row.at[0, 'Variant'])
    city_value = str(row.at[0, 'City'])
    bodytype_value = str(row.at[0, 'BodyType'])
    fuel_value = str(row.at[0, 'Fuel'])
    transmission_value = str(row.at[0, 'Transmission'])
    car_age_value = float(car_age)
    kms_per_year_value = float(kms_per_year)
    ownership_value = float(ownership)
    reg_state_value = None if resolved_state == 'UNK' else resolved_state

    composite, breakdown = adjuster.compute(
        make=make_value,
        model=model_value,
        variant=variant_value,
        city=city_value,
        bodytype=bodytype_value,
        fuel=fuel_value,
        transmission=transmission_value,
        car_age=car_age_value,
        kms_per_year=kms_per_year_value,
        ownership=ownership_value,
        reg_state=reg_state_value,
    )

    adjusted_price = base_price * composite

    return {
        'base_price': base_price,
        'adjusted_price': adjusted_price,
        'composite_multiplier': composite,
        'car_age': car_age,
        'kms_per_year': kms_per_year,
        'breakdown': breakdown,
    }


def _run_batch(uploaded_file):
    file_name = uploaded_file.name
    _, ext = os.path.splitext(file_name)
    ext = ext.lower()
    if ext not in ('.csv', '.xlsx', '.xls'):
        raise ValueError('Please upload a CSV or Excel file.')

    with tempfile.TemporaryDirectory() as tmp_dir:
        input_path = os.path.join(tmp_dir, file_name)
        with open(input_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())

        output_path = os.path.join(tmp_dir, f"{os.path.splitext(file_name)[0]}_predictions{ext}")

        stdout_buffer = io.StringIO()
        with redirect_stdout(stdout_buffer):
            predict_batch(input_path=input_path, output_path=output_path)

        output_df = _read_any(output_path)
        logs = stdout_buffer.getvalue()

        if ext == '.csv':
            output_bytes = output_df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
            mime = 'text/csv'
        else:
            excel_buffer = io.BytesIO()
            output_df.to_excel(excel_buffer, index=False)
            output_bytes = excel_buffer.getvalue()
            mime = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'

        output_file_name = os.path.basename(output_path)
        return output_df, output_bytes, output_file_name, logs, mime


def _extract_batch_metrics(logs: str) -> pd.DataFrame:
    # Parse the metrics already printed by src/batch_predict.py into a table for UI display.
    pattern = re.compile(
        r"---\s*(?P<label>.+?)\s*---\s*"
        r"Rows\s*:\s*(?P<rows>[\d,]+)\s*"
        r"MSE\s*:\s*(?P<mse>[\d,\.]+)\s*"
        r"RMSE\s*:\s*(?P<rmse>[\d,\.]+)\s*"
        r"(?:R²|R2)\s*:\s*(?P<r2>[\-\d\.]+)\s*"
        r"MAPE\s*:\s*(?P<mape>[\d\.]+)%",
        flags=re.MULTILINE,
    )

    rows = []
    for match in pattern.finditer(logs):
        rows.append(
            {
                'Label': match.group('label').strip(),
                'Rows': int(match.group('rows').replace(',', '')),
                'MSE': float(match.group('mse').replace(',', '')),
                'RMSE': float(match.group('rmse').replace(',', '')),
                'R-square': float(match.group('r2')),
                'MAPE (%)': float(match.group('mape')),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Guard against duplicate metric blocks in logs (e.g., repeated execution text).
    df = df.drop_duplicates(subset=['Label'], keep='last').copy()

    # Keep a stable, user-friendly order when both are present.
    order = {'Base Prediction': 0, 'Adjusted Prediction': 1}
    df['_order'] = df['Label'].map(order).fillna(999).astype(int)
    df = df.sort_values('_order').drop(columns=['_order']).reset_index(drop=True)
    return df


def _extract_log_sections(logs: str):
    normalized_logs = logs.replace('\r\n', '\n').replace('\r', '\n')
    # Match only real titled section headers, not separator lines like '-----...'.
    header_pattern = re.compile(r"^---\s*(?P<title>[A-Za-z].+?)\s*---\s*$", flags=re.MULTILINE)
    headers = list(header_pattern.finditer(normalized_logs))

    sections = []
    for i, match in enumerate(headers):
        title = match.group('title').strip()
        body_start = match.end()
        body_end = headers[i + 1].start() if i + 1 < len(headers) else len(normalized_logs)
        body = normalized_logs[body_start:body_end].strip()
        sections.append({'title': title, 'body': body})
    return sections


def _parse_segment_section_to_df(title: str, body: str) -> pd.DataFrame:
    rows = []
    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line or ':' not in line:
            if not (title.startswith('Segment Error Summary') and '|' in line):
                continue

        if title.startswith('Segment Error Summary'):
            if '|' not in line:
                continue
            parts = [p.strip() for p in line.split('|')]
            if len(parts) != 4:
                continue
            if parts[0].lower() in ('segment', 'make'):
                continue
            if set(parts[0]) == {'-'}:
                continue

            m_mape = re.search(r"([\-+]?\d+(?:\.\d+)?)%", parts[1])
            m_mpe = re.search(r"([\-+]?\d+(?:\.\d+)?)%", parts[2])
            if not m_mape or not m_mpe:
                continue

            rows.append(
                {
                    'Segment': parts[0],
                    'MAPE (%)': float(m_mape.group(1)),
                    'MPE (%)': float(m_mpe.group(1)),
                    'Direction': parts[3],
                }
            )
            continue

        if title.startswith('Segment MAPE'):
            m = re.match(r"^(?P<segment>.+?):\s*(?P<value>[\-+]?\d+(?:\.\d+)?)%$", line)
            if not m:
                continue
            rows.append(
                {
                    'Segment': m.group('segment').strip(),
                    'MAPE (%)': float(m.group('value')),
                }
            )
        elif title.startswith('Segment Bias'):
            m = re.match(
                r"^(?P<segment>.+?):\s*MPE\s*(?P<value>[\-+]?\d+(?:\.\d+)?)%\s*->\s*(?P<direction>.+)$",
                line,
            )
            if not m:
                continue
            rows.append(
                {
                    'Segment': m.group('segment').strip(),
                    'MPE (%)': float(m.group('value')),
                    'Direction': m.group('direction').strip(),
                }
            )

    return pd.DataFrame(rows)


def _build_segment_tables_from_output(output_df: pd.DataFrame):
    required_cols = {'Price (₹)', 'Base Predicted Price', 'Predicted Price'}
    if not required_cols.issubset(set(output_df.columns)):
        return []

    work = output_df.copy()
    work['actual'] = pd.to_numeric(work['Price (₹)'], errors='coerce')
    work['base_pred'] = pd.to_numeric(work['Base Predicted Price'], errors='coerce')
    work['adj_pred'] = pd.to_numeric(work['Predicted Price'], errors='coerce')
    work['Make'] = work.get('Make', 'Unknown')
    work['City'] = work.get('City', 'Unknown')
    work['Transmission'] = work.get('Transmission', 'Unknown')
    work['Fuel'] = work.get('Fuel', 'Unknown')
    work['BodyType'] = work.get('BodyType', 'Unknown')
    kms_col = work['KMs Driven'] if 'KMs Driven' in work.columns else pd.Series(np.nan, index=work.index)
    ownership_col = work['Ownership'] if 'Ownership' in work.columns else pd.Series(np.nan, index=work.index)
    year_col = work['Year'] if 'Year' in work.columns else pd.Series(np.nan, index=work.index)
    work['KMs Driven'] = pd.to_numeric(kms_col, errors='coerce')
    work['Ownership'] = pd.to_numeric(ownership_col, errors='coerce')
    work['Year'] = pd.to_numeric(year_col, errors='coerce')
    work['Make'] = work['Make'].fillna('Unknown').astype(str)
    work['City'] = work['City'].fillna('Unknown').astype(str)
    work['Transmission'] = work['Transmission'].fillna('Unknown').astype(str)
    work['Fuel'] = work['Fuel'].fillna('Unknown').astype(str)
    work['BodyType'] = work['BodyType'].fillna('Unknown').astype(str)

    valid_base = work['actual'].notna() & work['base_pred'].notna() & (work['actual'] != 0)
    valid_adj = work['actual'].notna() & work['adj_pred'].notna() & (work['actual'] != 0)

    bins = [-np.inf, 300000, 700000, 1200000, 2000000, np.inf]
    names = ['<=3L', '3-7L', '7-12L', '12-20L', '>20L']

    def _make_tables(mask: pd.Series, pred_col: str, label: str):
        if not mask.any():
            return []

        d = work.loc[mask, ['Make', 'actual', pred_col]].copy()
        d['ape'] = ((d[pred_col] - d['actual']).abs() / d['actual']) * 100
        d['mpe'] = ((d[pred_col] - d['actual']) / d['actual']) * 100
        d['price_band'] = pd.cut(d['actual'], bins=bins, labels=names)

        band_df = d.groupby('price_band', observed=True).agg(
            **{'MAPE (%)': ('ape', 'mean'), 'MPE (%)': ('mpe', 'mean')}
        ).reset_index(names='Segment')
        band_df['Direction'] = band_df['MPE (%)'].apply(
            lambda x: 'POSITIVE(over)' if x > 0 else ('NEGATIVE(under)' if x < 0 else 'NEUTRAL')
        )

        top_makes = d['Make'].value_counts().head(10).index
        make_df = d[d['Make'].isin(top_makes)].groupby('Make').agg(
            **{'MAPE (%)': ('ape', 'mean'), 'MPE (%)': ('mpe', 'mean')}
        ).sort_values(by='MAPE (%)').reset_index(names='Segment')
        make_df['Direction'] = make_df['MPE (%)'].apply(
            lambda x: 'POSITIVE(over)' if x > 0 else ('NEGATIVE(under)' if x < 0 else 'NEUTRAL')
        )

        year_df = d.join(work['Year'], how='left').dropna(subset=['Year']).copy()
        year_df['Year'] = year_df['Year'].astype(int)
        year_df = year_df.groupby('Year').agg(
            **{'MAPE (%)': ('ape', 'mean'), 'MPE (%)': ('mpe', 'mean')}
        ).sort_index().reset_index(names='Segment')
        year_df['Direction'] = year_df['MPE (%)'].apply(
            lambda x: 'POSITIVE(over)' if x > 0 else ('NEGATIVE(under)' if x < 0 else 'NEUTRAL')
        )

        city_d = work.loc[mask, ['City', 'actual', pred_col]].copy()
        city_d['ape'] = ((city_d[pred_col] - city_d['actual']).abs() / city_d['actual']) * 100
        city_d['mpe'] = ((city_d[pred_col] - city_d['actual']) / city_d['actual']) * 100
        top_cities = city_d['City'].value_counts().head(10).index
        city_df = city_d[city_d['City'].isin(top_cities)].groupby('City').agg(
            **{'MAPE (%)': ('ape', 'mean'), 'MPE (%)': ('mpe', 'mean')}
        ).sort_values(by='MAPE (%)').reset_index(names='Segment')
        city_df['Direction'] = city_df['MPE (%)'].apply(
            lambda x: 'POSITIVE(over)' if x > 0 else ('NEGATIVE(under)' if x < 0 else 'NEUTRAL')
        )

        kms_d = work.loc[mask, ['KMs Driven', 'actual', pred_col]].dropna(subset=['KMs Driven']).copy()
        kms_d['ape'] = ((kms_d[pred_col] - kms_d['actual']).abs() / kms_d['actual']) * 100
        kms_d['mpe'] = ((kms_d[pred_col] - kms_d['actual']) / kms_d['actual']) * 100
        kms_bins = [-np.inf, 20_000, 40_000, 60_000, 100_000, 150_000, np.inf]
        kms_names = ['<=20k', '20k-40k', '40k-60k', '60k-100k', '100k-150k', '>150k']
        kms_d['kms_band'] = pd.cut(kms_d['KMs Driven'], bins=kms_bins, labels=kms_names)
        kms_df = kms_d.groupby('kms_band', observed=True).agg(
            **{'MAPE (%)': ('ape', 'mean'), 'MPE (%)': ('mpe', 'mean')}
        ).reset_index(names='Segment')
        kms_df['Direction'] = kms_df['MPE (%)'].apply(
            lambda x: 'POSITIVE(over)' if x > 0 else ('NEGATIVE(under)' if x < 0 else 'NEUTRAL')
        )

        own_d = work.loc[mask, ['Ownership', 'actual', pred_col]].dropna(subset=['Ownership']).copy()
        own_d['ape'] = ((own_d[pred_col] - own_d['actual']).abs() / own_d['actual']) * 100
        own_d['mpe'] = ((own_d[pred_col] - own_d['actual']) / own_d['actual']) * 100
        own_bins = [0, 1, 2, 3, 4, np.inf]
        own_names = ['1', '2', '3', '4', '5+']
        own_d['ownership_band'] = pd.cut(own_d['Ownership'], bins=own_bins, labels=own_names, include_lowest=True)
        ownership_df = own_d.groupby('ownership_band', observed=True).agg(
            **{'MAPE (%)': ('ape', 'mean'), 'MPE (%)': ('mpe', 'mean')}
        ).reset_index(names='Segment')
        ownership_df['Direction'] = ownership_df['MPE (%)'].apply(
            lambda x: 'POSITIVE(over)' if x > 0 else ('NEGATIVE(under)' if x < 0 else 'NEUTRAL')
        )

        tx_d = work.loc[mask, ['Transmission', 'actual', pred_col]].copy()
        tx_d['ape'] = ((tx_d[pred_col] - tx_d['actual']).abs() / tx_d['actual']) * 100
        tx_d['mpe'] = ((tx_d[pred_col] - tx_d['actual']) / tx_d['actual']) * 100
        transmission_df = tx_d.groupby('Transmission').agg(
            **{'MAPE (%)': ('ape', 'mean'), 'MPE (%)': ('mpe', 'mean')}
        ).sort_values(by='MAPE (%)').reset_index(names='Segment')
        transmission_df['Direction'] = transmission_df['MPE (%)'].apply(
            lambda x: 'POSITIVE(over)' if x > 0 else ('NEGATIVE(under)' if x < 0 else 'NEUTRAL')
        )

        fuel_d = work.loc[mask, ['Fuel', 'actual', pred_col]].copy()
        fuel_d['ape'] = ((fuel_d[pred_col] - fuel_d['actual']).abs() / fuel_d['actual']) * 100
        fuel_d['mpe'] = ((fuel_d[pred_col] - fuel_d['actual']) / fuel_d['actual']) * 100
        fuel_df = fuel_d.groupby('Fuel').agg(
            **{'MAPE (%)': ('ape', 'mean'), 'MPE (%)': ('mpe', 'mean')}
        ).sort_values(by='MAPE (%)').reset_index(names='Segment')
        fuel_df['Direction'] = fuel_df['MPE (%)'].apply(
            lambda x: 'POSITIVE(over)' if x > 0 else ('NEGATIVE(under)' if x < 0 else 'NEUTRAL')
        )

        bt_d = work.loc[mask, ['BodyType', 'actual', pred_col]].copy()
        bt_d['ape'] = ((bt_d[pred_col] - bt_d['actual']).abs() / bt_d['actual']) * 100
        bt_d['mpe'] = ((bt_d[pred_col] - bt_d['actual']) / bt_d['actual']) * 100
        bodytype_df = bt_d.groupby('BodyType').agg(
            **{'MAPE (%)': ('ape', 'mean'), 'MPE (%)': ('mpe', 'mean')}
        ).sort_values(by='MAPE (%)').reset_index(names='Segment')
        bodytype_df['Direction'] = bodytype_df['MPE (%)'].apply(
            lambda x: 'POSITIVE(over)' if x > 0 else ('NEGATIVE(under)' if x < 0 else 'NEUTRAL')
        )

        return [
            {'title': f'Segment Error Summary ({label}) by Price Band', 'df': band_df},
            {'title': f'Segment Error Summary ({label}) by Top Makes', 'df': make_df},
            {'title': f'Segment Error Summary ({label}) by Manufacturing Year', 'df': year_df},
            {'title': f'Segment Error Summary ({label}) by Top Cities', 'df': city_df},
            {'title': f'Segment Error Summary ({label}) by KMs Driven Range', 'df': kms_df},
            {'title': f'Segment Error Summary ({label}) by Ownership', 'df': ownership_df},
            {'title': f'Segment Error Summary ({label}) by Transmission', 'df': transmission_df},
            {'title': f'Segment Error Summary ({label}) by Fuel', 'df': fuel_df},
            {'title': f'Segment Error Summary ({label}) by BodyType', 'df': bodytype_df},
        ]

    return _make_tables(valid_base, 'base_pred', 'Base') + _make_tables(valid_adj, 'adj_pred', 'Adjusted')


st.set_page_config(page_title='Used Car Price Predictor', layout='wide')
st.title('Used Car Price Predictor')
st.caption('Single-car estimate and batch file prediction using your trained model artifacts.')

single_tab, batch_tab = st.tabs(['Single Prediction', 'Batch Prediction'])

with single_tab:
    st.subheader('Predict One Car')

    c1, c2, c3 = st.columns(3)
    with c1:
        city = st.text_input('City', value='Bangalore')
        make = st.text_input('Make', value='Maruti')
        model_name = st.text_input('Model', value='Swift')
        variant = st.text_input('Variant', value='VXI')
    with c2:
        transmission = st.selectbox('Transmission', options=['Manual', 'Automatic'])
        fuel = st.text_input('Fuel', value='Petrol')
        bodytype = st.text_input('BodyType', value='Hatchback')
        year = st.number_input('Year', min_value=1995, max_value=datetime.now().year, value=2018, step=1)
    with c3:
        ownership = st.number_input('Ownership', min_value=1.0, max_value=10.0, value=1.0, step=1.0)
        kms_driven = st.number_input('KMs Driven', min_value=0.0, value=45000.0, step=1000.0)
        registration = st.text_input('Registration (optional)', value='')
        reg_state = st.text_input('Reg State override (optional, e.g. KA)', value='')

    if st.button('Predict Price', type='primary'):
        try:
            result = _single_predict(
                city=city,
                make=make,
                model_name=model_name,
                variant=variant,
                transmission=transmission,
                fuel=fuel,
                bodytype=bodytype,
                year=int(year),
                ownership=float(ownership),
                kms_driven=float(kms_driven),
                registration=registration,
                reg_state=reg_state,
            )

            m1, m2, m3 = st.columns(3)
            m1.metric('Base Price', f"INR {result['base_price']:,.0f}")
            m2.metric('Demand Adjusted Price', f"INR {result['adjusted_price']:,.0f}")
            m3.metric('Demand Multiplier', f"{result['composite_multiplier']:.3f}x")

            st.write(
                f"Derived Car Age: {result['car_age']} years | Derived KMs/Year: {result['kms_per_year']:,.0f}"
            )

            breakdown_df = pd.DataFrame(
                result['breakdown'], columns=['Component', 'Multiplier', 'Weight']
            )
            st.dataframe(breakdown_df, use_container_width=True)
        except Exception as e:
            st.error(f'Single prediction failed: {e}')

with batch_tab:
    st.subheader('Predict from CSV or Excel')
    uploaded = st.file_uploader('Upload input file', type=['csv', 'xlsx', 'xls'])

    if uploaded is not None:
        st.write(f"Uploaded: {uploaded.name}")

        if st.button('Run Batch Prediction', type='primary'):
            try:
                output_df, output_bytes, output_name, logs, mime = _run_batch(uploaded)

                st.success('Batch prediction completed.')

                metrics_df = _extract_batch_metrics(logs)
                if not metrics_df.empty:
                    st.subheader('Batch Metrics')
                    st.dataframe(metrics_df, use_container_width=True)

                sections = _extract_log_sections(logs)
                segment_sections = [
                    s
                    for s in sections
                    if s['title'].startswith('Segment MAPE')
                    or s['title'].startswith('Segment Bias')
                    or s['title'].startswith('Segment Error Summary')
                ]
                if segment_sections:
                    st.subheader('Segment Metrics')
                    for section in segment_sections:
                        st.markdown(f"**{section['title']}**")
                        section_df = _parse_segment_section_to_df(section['title'], section['body'])
                        if not section_df.empty:
                            st.dataframe(section_df, use_container_width=True)
                        else:
                            st.text(section['body'])
                else:
                    fallback_tables = _build_segment_tables_from_output(output_df)
                    if fallback_tables:
                        st.subheader('Segment Metrics')
                        st.caption('Derived from output rows because segment log parsing returned no sections.')
                        for item in fallback_tables:
                            st.markdown(f"**{item['title']}**")
                            st.dataframe(item['df'], use_container_width=True)

                st.subheader('Batch Output Rows')
                st.dataframe(output_df, use_container_width=True)

                st.download_button(
                    label='Download Predicted File',
                    data=output_bytes,
                    file_name=output_name,
                    mime=mime,
                )

                st.subheader('Full Batch Script Output')
                st.text_area('Logs', value=logs or 'No logs captured.', height=360)
            except Exception as e:
                st.error(f'Batch prediction failed: {e}')
