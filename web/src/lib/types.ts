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
