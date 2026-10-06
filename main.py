import uvicorn


def main() -> None:
    """Start the DataPilot API for local development."""
    uvicorn.run("app.api.main:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
