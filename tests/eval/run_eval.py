import os
import time
import json
import argparse
import requests
import yaml
from collections import defaultdict
from pathlib import Path

def evaluate():
    parser = argparse.ArgumentParser(description="Run evaluation suite for DataPilot AI.")
    parser.add_argument("--dataset", type=str, help="Run only one specific dataset")
    parser.add_argument("--api-url", type=str, default=os.getenv("API_URL", "http://localhost:8000"), help="DataPilot API base URL")
    args = parser.parse_args()

    api_url = args.api_url.rstrip("/")
    
    base_dir = Path(__file__).resolve().parent
    questions_file = base_dir / "questions.yaml"
    results_file = base_dir / "results.json"
    data_dir = base_dir.parent.parent / "data"

    with open(questions_file, "r") as f:
        config = yaml.safe_load(f)

    all_results = []
    
    datasets = config.get("datasets", {})
    if args.dataset:
        if args.dataset in datasets:
            datasets = {args.dataset: datasets[args.dataset]}
        else:
            print(f"Dataset {args.dataset} not found in questions.yaml")
            return

    overall_stats = {"total": 0, "passed": 0, "failed": 0}
    category_stats = defaultdict(lambda: {"total": 0, "passed": 0})
    failure_causes = defaultdict(list)

    for dataset_name, dataset_info in datasets.items():
        print(f"\nEvaluating dataset: {dataset_name}")
        csv_path = data_dir / f"{dataset_name}.csv"
        
        # Upload CSV
        table_name = dataset_name
        if csv_path.exists():
            print(f"Uploading {csv_path} to {api_url}/upload ...")
            try:
                with open(csv_path, "rb") as f:
                    upload_res = requests.post(f"{api_url}/upload", files={"file": (csv_path.name, f, "text/csv")}, timeout=45)
                if upload_res.status_code == 200:
                    table_name = upload_res.json().get("table", dataset_name)
                    print(f"Uploaded successfully as table '{table_name}'")
                else:
                    print(f"Upload failed: HTTP {upload_res.status_code} - {upload_res.text}")
            except Exception as e:
                print(f"Failed to upload CSV: {e}")
        else:
            print(f"Warning: CSV file not found at {csv_path}, proceeding anyway.")

        questions = dataset_info.get("questions", [])
        for q in questions:
            q_text = q["text"]
            category = q["category"]
            expected_answerable = q.get("answerable", True)
            
            print(f"  Q: {q_text}")
            
            start_time = time.time()
            error_msg = None
            sql_hint = ""
            executed = False
            returned_rows = False
            failure_category = "success"
            
            try:
                payload = {"question": q_text, "history": []}
                res = requests.post(f"{api_url}/query", json=payload, timeout=120)
                
                if res.status_code == 200:
                    result = res.json()
                    plan = result.get("plan", {})
                    sql_hint = plan.get("sql_hint", "")
                    error_msg = result.get("error")
                    rows = result.get("results", [])
                    returned_rows = len(rows) > 0
                    
                    if error_msg:
                        if "syntax" in error_msg.lower() or "parser" in error_msg.lower():
                            failure_category = "syntax_error"
                        elif "column" in error_msg.lower() or "binder" in error_msg.lower():
                            failure_category = "wrong_column"
                        else:
                            failure_category = "other_error"
                    else:
                        executed = True
                        if not expected_answerable:
                            # For unanswerable questions, success means acknowledging it's unanswerable or empty/guarded
                            answer = result.get("answer", "").lower()
                            if not returned_rows or "cannot" in answer or "unanswerable" in answer or "not available" in answer:
                                failure_category = "unanswerable_correct"
                            else:
                                failure_category = "success"
                        else:
                            if not returned_rows:
                                failure_category = "empty_result"
                            else:
                                failure_category = "success"
                else:
                    error_msg = f"HTTP {res.status_code}: {res.text}"
                    err_lower = error_msg.lower()
                    if "syntax" in err_lower or "parse" in err_lower:
                        failure_category = "syntax_error"
                    elif "column" in err_lower or "binder" in err_lower or "refer to" in err_lower:
                        failure_category = "wrong_column"
                    elif "timeout" in err_lower:
                        failure_category = "timeout"
                    else:
                        failure_category = "other_error"
                    
            except requests.exceptions.Timeout:
                error_msg = "Timeout"
                failure_category = "timeout"
            except Exception as e:
                error_msg = str(e)
                failure_category = "other_error"
                
            latency = time.time() - start_time
            
            passed = failure_category in ("success", "unanswerable_correct")
            
            record = {
                "dataset": dataset_name,
                "question": q_text,
                "category": category,
                "sql_hint": sql_hint,
                "executed": executed,
                "returned_rows": returned_rows,
                "error": error_msg,
                "latency": latency,
                "failure_category": failure_category,
                "passed": passed
            }
            all_results.append(record)
            
            overall_stats["total"] += 1
            category_stats[category]["total"] += 1
            if passed:
                overall_stats["passed"] += 1
                category_stats[category]["passed"] += 1
            else:
                overall_stats["failed"] += 1
                failure_causes[failure_category].append(q_text)
                
            print(f"     -> {failure_category} ({latency:.2f}s)")

    # Save results
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
        
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    
    pass_rate = (overall_stats["passed"] / overall_stats["total"] * 100) if overall_stats["total"] > 0 else 0
    print(f"Overall Pass Rate: {pass_rate:.1f}% ({overall_stats['passed']}/{overall_stats['total']})")
    
    print("\nPass Rate by Category:")
    print("| Category | Pass Rate | Passed / Total |")
    print("|----------|-----------|----------------|")
    for cat, stats in sorted(category_stats.items()):
        cat_rate = (stats["passed"] / stats["total"] * 100) if stats["total"] > 0 else 0
        print(f"| {cat} | {cat_rate:.1f}% | {stats['passed']} / {stats['total']} |")
        
    print("\nTop Failure Causes:")
    for cause, qs in sorted(failure_causes.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"- {cause}: {len(qs)} failures")
        print(f"  Example: '{qs[0]}'")
        
    print(f"\nResults saved to {results_file}")

if __name__ == "__main__":
    evaluate()
