export const ALLOWED_CC_BANKS = new Set([
  "HDFC Bank", "State Bank of India", "ICICI Bank", "Axis Bank",
  "Kotak Mahindra Bank", "RBL Bank", "IDFC First Bank",
  "IndusInd Bank", "Bank of Baroda", "Yes Bank", "Canara Bank",
  "HSBC", "_RESIDUAL",
]);

export const ALLOWED_DC_BANKS = new Set([
  "State Bank of India", "Bank of Baroda", "Canara Bank", "HDFC Bank",
  "Union Bank of India", "Punjab National Bank", "Axis Bank",
  "Bank of India", "Kotak Mahindra Bank", "Indian Bank",
  "ICICI Bank", "Paytm Payments Bank", "Central Bank of India",
  "India Post Payments Bank", "Indian Overseas Bank", "UCO Bank",
  "_RESIDUAL",
]);

export function displayBank(name: string): string {
  if (name === "_RESIDUAL") return "All Other Banks (Residual)";
  return name;
}
