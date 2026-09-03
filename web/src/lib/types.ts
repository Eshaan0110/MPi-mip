export interface BankForecast {
  bank_name: string;
  card_type: "CC" | "DC";
  forecast_month: string;
  yhat: number;
  yhat_lower: number | null;
  yhat_upper: number | null;
  model_type: string | null;
}

export interface AggregateForecast {
  metric: string;
  forecast_month: string;
  yhat: number;
  yhat_lower: number | null;
  yhat_upper: number | null;
  model_type: string | null;
}

export interface ProcessedAggregate {
  metric: string;
  month: string;
  value: number;
}

export interface ProcessedBankSeries {
  bank_name: string;
  card_type: "CC" | "DC";
  month: string;
  y: number;
}

export interface ModelMetadata {
  bank_name: string | null;
  card_type: string | null;
  metric: string | null;
  model_type: string;
  cv_mape: number | null;
  oos_mape: number | null;
  last_trained: string;
}

export interface ScraperRun {
  id: string;
  source: string;
  started_at: string;
  completed_at: string | null;
  status: "running" | "success" | "failed" | "partial";
  files_downloaded: number;
  records_written: number;
  error_message: string | null;
}

export interface DataStatus {
  source: string;
  last_success: string | null;
  last_run: string | null;
  status: string;
  latest_data_month: string | null;
}

export interface Scenario {
  forecast_month: string;
  scenario: string;
  repo_rate: number | null;
  label: string | null;
  yhat: number;
  yhat_prophet: number | null;
  yhat_arima: number | null;
  yhat_arimax: number | null;
  yhat_ets: number | null;
  yhat_direct: number | null;
}

export interface ScorecardScore {
  model_name: string;
  forecast_month: string;
  forecast_value: number;
  actual_value: number;
  ape: number;
}
