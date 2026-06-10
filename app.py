"""
VClinic GraphRAG — CLI entrypoint

Commands:
  build   — Ingest CSV data + generate embeddings into Neo4j
  chat    — Interactive chat loop with the GraphRAG agent
  ask     — Ask a single question (non-interactive)
"""

import typer
from rich.console import Console
from rich.panel import Panel

app = typer.Typer(help="VClinic GraphRAG CLI")
console = Console()


@app.command()
def build(
    data_dir: str = typer.Option(
        "/Users/yunwen/work/test_data/vclinic",
        "--data-dir", "-d",
        help="Path to the folder containing VClinic CSV files.",
    )
):
    """Ingest VClinic CSV data into Neo4j and generate vector embeddings."""
    from src.pipeline.build_graph import build_graph
    build_graph(data_dir)


@app.command()
def chat():
    """Start an interactive chat session with the VClinic GraphRAG agent."""
    from src.agent.graph_analysis_agent import ask

    console.print(Panel(
        "[bold cyan]VClinic GraphRAG Agent[/bold cyan]\n"
        "Ask clinical questions about patients, conditions, medications, and more.\n"
        "Type [bold]exit[/bold] or [bold]quit[/bold] to stop.",
        expand=False,
    ))

    while True:
        try:
            question = console.input("\n[bold green]You:[/bold green] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye.[/dim]")
            break

        if question.lower() in {"exit", "quit", "q"}:
            console.print("[dim]Goodbye.[/dim]")
            break

        if not question:
            continue

        console.print("\n[bold blue]Agent:[/bold blue] ", end="")
        try:
            answer = ask(question)
            console.print(answer)
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")


@app.command()
def ask_once(
    question: str = typer.Argument(..., help="The clinical question to ask."),
):
    """Ask a single question and print the answer."""
    from src.agent.graph_analysis_agent import ask

    try:
        answer = ask(question)
        console.print(answer)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
