// k6 load test for the CreditIQ scoring API.
//
//   k6 run -e BASE_URL=http://localhost:8000 -e API_KEY=dev-analyst-key-change-me \
//          -e SCENARIO=latency --summary-export loadtest/results/latency.json loadtest/score.js
//
// SCENARIO=latency    open model: 20 requests/s to /v1/score for 60 s (the p95 < 50 ms target)
// SCENARIO=decide     open model: 20 requests/s to /v1/decide for 60 s (policy rules + full explanation)
// SCENARIO=throughput closed model: 8 virtual users back-to-back on /v1/score for 60 s (capacity)
//
// The applicants are SYNTHETIC rows (loadtest/applicants.json). Raise the API's RATE_LIMIT_SCORE for the
// throughput run, otherwise the per-key limit (default 600/min) answers 429 by design.
import http from "k6/http";
import { check } from "k6";
import { SharedArray } from "k6/data";

const applicants = new SharedArray("applicants", () => JSON.parse(open("./applicants.json")));
const BASE = __ENV.BASE_URL || "http://localhost:8000";
const KEY = __ENV.API_KEY || "dev-analyst-key-change-me";
const SCENARIO = __ENV.SCENARIO || "latency";

const scenarios = {
  latency: { executor: "constant-arrival-rate", rate: 20, timeUnit: "1s", duration: "60s", preAllocatedVUs: 10, maxVUs: 40 },
  decide: { executor: "constant-arrival-rate", rate: 20, timeUnit: "1s", duration: "60s", preAllocatedVUs: 10, maxVUs: 40 },
  throughput: { executor: "constant-vus", vus: 8, duration: "60s" },
};

export const options = {
  scenarios: { [SCENARIO]: { ...scenarios[SCENARIO], startTime: "0s" } },
  summaryTrendStats: ["avg", "min", "med", "p(90)", "p(95)", "p(99)", "max"],
  thresholds: {
    http_req_failed: ["rate<0.01"],
    ...(SCENARIO === "latency" ? { http_req_duration: ["p(95)<50"] } : {}),
  },
};

export function setup() {
  const r = http.get(`${BASE}/readyz`);
  if (r.status !== 200) throw new Error(`API not ready: ${r.status}`);
  // warm-up: the first requests pay for lazy imports and caches
  for (let i = 0; i < 20; i++) post(applicants[i % applicants.length]);
}

const path = SCENARIO === "decide" ? "/v1/decide" : "/v1/score";

function post(applicant) {
  return http.post(`${BASE}${path}`, JSON.stringify({ applicant }), {
    headers: { "Content-Type": "application/json", "X-API-Key": KEY },
    tags: { endpoint: path },
  });
}

export default function () {
  const res = post(applicants[Math.floor(Math.random() * applicants.length)]);
  check(res, { "status 200": (r) => r.status === 200, "has pd": (r) => r.status === 200 && r.json("pd") !== undefined });
}
