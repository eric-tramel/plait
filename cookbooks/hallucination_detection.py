#!/usr/bin/env python3
"""Hallucination Detection Training with HaluBench.

This cookbook demonstrates training a hallucination detection system using
the HaluBench dataset from Hugging Face. The system learns to identify
whether an answer contains hallucinations given a context and question.

Pipeline Architecture:
    (context, question, answer)
            |
            v
    HallucinationDetector <- Learnable system prompt
            |
            v
    true/false prediction
            |
            v
    Accuracy Loss <- Evaluates correctness
            |
            v
    Backward Pass <- Updates system prompt

Key features demonstrated:
- Training with real HuggingFace dataset (HaluBench)
- Batch processing with batch size 32
- Binary classification (true/false) output
- Accuracy-based optimization
- GPT-4o-mini for forward pass, GPT-5.2 for optimization
- Rich progress visualization

Requirements:
    export OPENAI_API_KEY=your-api-key

Run with:
    python cookbooks/hallucination_detection.py
"""

import asyncio
import os
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from datasets import load_dataset
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text

from inf_engine.execution.context import ExecutionSettings
from inf_engine.module import InferenceModule, LLMInference
from inf_engine.optimization import (
    SFAOptimizer,
    VerifierLoss,
)
from inf_engine.optimization.backward import BackwardResult
from inf_engine.optimization.feedback import Feedback
from inf_engine.parameter import Parameter
from inf_engine.resources.config import (
    OpenAIEndpointConfig,
    ResourceConfig,
)

# Initialize rich console
console = Console()


# =============================================================================
# Template Parameter with Traceable Formatting
# =============================================================================


class TemplateParameter(Parameter):
    """A learnable template string with named placeholders.

    Extends Parameter to support template formatting with automatic
    backward pass feedback generation via `as_module()`.

    Args:
        value: The template string with {placeholder} format specifiers.
        placeholders: List of placeholder names expected in the template.
        description: Description for the optimizer.
        requires_grad: Whether this parameter should be optimized.

    Example:
        >>> template = TemplateParameter(
        ...     "Context: {context}\\nQuestion: {question}",
        ...     placeholders=["context", "question"],
        ...     description="Formats the input for the LLM",
        ... )
        >>> formatter = template.as_module()
        >>> # Use formatter in forward() to get traced formatting
    """

    def __init__(
        self,
        value: str,
        placeholders: list[str],
        description: str,
        requires_grad: bool = True,
    ) -> None:
        super().__init__(
            value=value,
            description=description,
            requires_grad=requires_grad,
        )
        self.placeholders = placeholders

    def as_module(self) -> "TemplateFormatter":
        """Return a module that formats this template.

        The returned module participates in tracing and generates
        appropriate backward feedback for the template parameter.

        Returns:
            A TemplateFormatter module wrapping this parameter.
        """
        return TemplateFormatter(self)


class TemplateFormatter(InferenceModule):
    """A module that formats a TemplateParameter with provided values.

    This module participates in tracing, allowing the template parameter
    to receive feedback during backward passes.

    Args:
        template: The TemplateParameter to format.

    Example:
        >>> template = TemplateParameter("Hello, {name}!", ["name"], "Greeting template")
        >>> formatter = TemplateFormatter(template)
        >>> # In forward(): formatted = formatter(name="World")
    """

    # Type annotation for the template attribute
    template: TemplateParameter

    def __init__(self, template: TemplateParameter) -> None:
        super().__init__()
        # Use object.__setattr__ to avoid registering the template as a
        # parameter of this module. The template is already registered
        # on the parent module that owns it.
        object.__setattr__(self, "template", template)

    def forward(self, **kwargs: str) -> str:
        """Format the template with provided values.

        Args:
            **kwargs: Values for each placeholder in the template.

        Returns:
            The formatted string.

        Raises:
            KeyError: If a required placeholder is missing from kwargs.
        """
        return self.template.value.format(**kwargs)

    async def backward(self, feedback: Any, ctx: Any) -> BackwardResult:
        """Generate feedback for the template parameter.

        Args:
            feedback: Feedback from downstream nodes.
            ctx: BackwardContext with inputs and output.

        Returns:
            BackwardResult with parameter feedback for the template.
        """
        result = BackwardResult()

        # Pass feedback through to inputs (the template variables)
        for input_name in ctx.inputs:
            result.input_feedback[input_name] = Feedback(
                content=f"Template formatting received feedback: {feedback.content}",
                score=feedback.score,
                feedback_type=feedback.feedback_type,
            )

        # Generate feedback for the template parameter if learnable
        if self.template.requires_grad:
            # Collect input values for context
            input_summary = ", ".join(
                f"{k}={repr(str(v)[:100])}" for k, v in ctx.inputs.items()
            )
            output_text = str(ctx.output)[:500]

            score_info = (
                f"Score: {feedback.score}" if feedback.score is not None else ""
            )

            result.parameter_feedback["template"] = f"""
The template parameter:
"{self.template.value}"

Parameter description: {self.template.description}
Expected placeholders: {self.template.placeholders}

Was formatted with inputs: {input_summary}

Produced output: {output_text}{"..." if len(str(ctx.output)) > 500 else ""}

Received this feedback: {feedback.content}
{score_info}

Suggest specific improvements to the template that would address this feedback.
The template must keep the placeholders: {{{", ".join(f"{{{p}}}" for p in self.template.placeholders)}}}
""".strip()

        return result


# =============================================================================
# Hallucination Detection Pipeline
# =============================================================================


class RegexExtractor(InferenceModule):
    """A module that extracts and normalizes verdicts using a learnable regex.

    This module is part of the traced graph, allowing the regex pattern
    to receive feedback and be optimized during training.
    """

    def __init__(self) -> None:
        super().__init__()

        # Fixed regex pattern for extracting the verdict (not learnable)
        self.pattern = Parameter(
            r"(true|false)",
            description=(
                "Regex pattern to extract the hallucination verdict from the LLM response. "
                "The pattern should capture a group that indicates whether the answer is "
                "hallucinated ('true') or correct ('false'). The first capturing group "
                "will be used."
            ),
            requires_grad=False,
        )

    def forward(self, response: str) -> str:
        """Extract and normalize the verdict from LLM response.

        Args:
            response: Raw LLM response text.

        Returns:
            "true" if hallucinated, "false" if correct, or "unknown" if
            extraction failed.
        """
        # Convert to string in case we get a TracedOutput
        response_str = str(response)

        try:
            match = re.search(self.pattern.value, response_str, re.IGNORECASE)

            if match:
                # Get the first capturing group, or the whole match if no groups
                extracted = match.group(1) if match.lastindex else match.group(0)
                extracted = extracted.strip().lower()

                # Normalize to true/false
                if extracted in ("true", "yes", "hallucinated", "1"):
                    return "true"
                elif extracted in ("false", "no", "not hallucinated", "0"):
                    return "false"

            # No match or unrecognized value
            return "unknown"

        except re.error:
            # Invalid regex pattern
            return "unknown"

    async def backward(self, feedback: Any, ctx: Any) -> BackwardResult:
        """Generate feedback for the regex pattern parameter.

        Args:
            feedback: Feedback from downstream nodes.
            ctx: BackwardContext with inputs and output.

        Returns:
            BackwardResult with parameter feedback for the pattern.
        """
        result = BackwardResult()

        # Pass feedback through to the input (the LLM response)
        result.input_feedback["response"] = Feedback(
            content=f"Regex extraction received feedback: {feedback.content}",
            score=feedback.score,
            feedback_type=feedback.feedback_type,
        )

        # Generate feedback for the pattern parameter if learnable
        if self.pattern.requires_grad:
            input_text = str(ctx.inputs.get("response", ""))[:500]
            output_text = str(ctx.output)

            score_info = (
                f"Score: {feedback.score}" if feedback.score is not None else ""
            )

            result.parameter_feedback["pattern"] = f"""
The regex pattern:
"{self.pattern.value}"

Parameter description: {self.pattern.description}

Was applied to input: {input_text}{"..." if len(str(ctx.inputs.get("response", ""))) > 500 else ""}

Extracted output: {output_text}

Received this feedback: {feedback.content}
{score_info}

Suggest specific improvements to the regex pattern that would address this feedback.
The pattern must be a valid Python regex and should capture true/false or equivalent values.
""".strip()

        return result


class HallucinationDetector(InferenceModule):
    """A hallucination detection module with learnable prompts and regex.

    Takes a context, question, and answer as input and determines whether
    the answer contains hallucinations (factual errors or unsupported claims).

    The module uses:
    1. A learnable system prompt to configure the LLM's behavior
    2. A learnable user prompt template to format the input
    3. A learnable regex pattern to extract the verdict from the response

    Output: "true" if the answer is hallucinated, "false" if it is correct.
    """

    def __init__(self) -> None:
        super().__init__()

        # Learnable system prompt for hallucination detection
        self.system_prompt = Parameter(
            (
                "You are a hallucination detector. Given a context passage, "
                "a question, and an answer, determine if the answer contains "
                "any hallucinations (factual errors or claims not supported "
                "by the context). Respond with only 'true' if hallucinated "
                "or 'false' if the answer is correct."
            ),
            description=(
                "System prompt for hallucination detection. Should instruct the LLM to "
                "carefully compare the answer against the context, identify any claims "
                "not supported by the context, detect factual inconsistencies, and "
                "output a clear true/false verdict. "
                "The prompt must not exceed 2048 tokens (8,192 characters). "
                "Output instructions must match Regex pattern."
            ),
        )

        # Learnable user prompt template (uses {context}, {question}, {answer} placeholders)
        self.user_prompt_template = TemplateParameter(
            (
                "Context:\n{context}\n\n"
                "Question:\n{question}\n\n"
                "Answer:\n{answer}\n\n"
                "Is this answer hallucinated?"
            ),
            placeholders=["context", "question", "answer"],
            description=(
                "User prompt template for formatting the input to the LLM. "
                "The template structures how the context, question, and answer are "
                "presented to the model. "
                "Keep it concise but complete."
            ),
        )

        # Template formatter module (participates in tracing for backward pass)
        self.formatter = self.user_prompt_template.as_module()

        self.detector = LLMInference(
            alias="detector",
            system_prompt=self.system_prompt,
            temperature=0.0,  # Deterministic for classification
            max_tokens=50,  # Allow more tokens for structured responses
        )

        # Regex extractor as a sub-module (part of the traced graph)
        self.extractor = RegexExtractor()

    def forward(self, context: str, question: str, answer: str) -> str:
        """Detect if the answer contains hallucinations.

        Args:
            context: The source passage/context.
            question: The question being answered.
            answer: The answer to evaluate for hallucinations.

        Returns:
            "true" if hallucinated, "false" if correct, "unknown" if
            extraction failed.
        """
        # Step 1: Format the input using the learnable template (traced)
        prompt = self.formatter(context=context, question=question, answer=answer)

        # Step 2: Get LLM response
        response = self.detector(prompt)

        # Step 3: Extract verdict using the regex extractor (traced)
        verdict = self.extractor(response)
        return verdict


# =============================================================================
# Accuracy-Based Loss Function
# =============================================================================


def make_accuracy_verifier(
    ground_truth: str,
) -> Callable[[str], tuple[bool, str]]:
    """Create a verifier that checks prediction accuracy.

    Args:
        ground_truth: The expected label ("true" or "false").

    Returns:
        A verifier function for use with VerifierLoss.
    """

    def verify(output: str) -> tuple[bool, str]:
        """Verify if prediction matches ground truth."""
        # The output is already the extracted verdict from the RegexExtractor
        prediction = str(output).strip().lower()

        # Handle extraction failure
        if prediction == "unknown":
            return False, "Failed to extract verdict from LLM response"

        # Compare with ground truth
        if prediction == ground_truth.lower():
            return True, f"Correct: predicted {prediction}"
        else:
            return False, f"Wrong: predicted {prediction}, expected {ground_truth}"

    return verify


# =============================================================================
# Data Loading
# =============================================================================


def load_halubench_data(
    batch_size: int = 32, val_size: int = 256
) -> tuple[list[list[dict]], list[dict]]:
    """Load HaluBench dataset and split into training batches and validation set.

    Args:
        batch_size: Number of samples per training batch.
        val_size: Number of samples to reserve for validation.

    Returns:
        Tuple of (training_batches, validation_samples).
    """
    with console.status("[bold blue]Loading HaluBench dataset from HuggingFace..."):
        dataset = load_dataset("PatronusAI/HaluBench", split="test")

    # Convert to list of dicts with normalized field names
    samples = []
    for item in dataset:
        # Map HaluBench label to true/false
        # HaluBench uses "FAIL" for hallucinated, "PASS" for correct
        label = item["label"]
        is_hallucinated = label.upper() == "FAIL"

        samples.append(
            {
                "context": item["passage"],
                "question": item["question"],
                "answer": item["answer"],
                "label": "true" if is_hallucinated else "false",
            }
        )

    # Split into validation and training sets
    # Take first val_size samples for validation (deterministic split)
    val_samples = samples[:val_size]
    train_samples = samples[val_size:]

    # Split training into batches
    train_batches = []
    for i in range(0, len(train_samples), batch_size):
        batch = train_samples[i : i + batch_size]
        train_batches.append(batch)

    console.print(
        f"[green]Loaded {len(samples)} total samples:[/green]\n"
        f"  [dim]Training: {len(train_samples)} samples in {len(train_batches)} batches[/dim]\n"
        f"  [dim]Validation: {len(val_samples)} samples[/dim]"
    )
    return train_batches, val_samples


# =============================================================================
# Validation
# =============================================================================


async def run_validation(
    detector: HallucinationDetector,
    val_samples: list[dict],
    batch_size: int = 32,
) -> float:
    """Run validation on the validation set.

    Args:
        detector: The hallucination detector module.
        val_samples: List of validation samples.
        batch_size: Batch size for validation inference.

    Returns:
        Validation accuracy as a float between 0 and 1.
    """
    correct = 0
    total = 0

    # Process in batches for efficiency
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold yellow]Validating..."),
        BarColumn(bar_width=30),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        console=console,
        transient=True,
    )

    with progress:
        val_task = progress.add_task("Validation", total=len(val_samples))

        for i in range(0, len(val_samples), batch_size):
            batch = val_samples[i : i + batch_size]

            # Run forward passes in parallel
            forward_tasks = [
                detector(
                    sample["context"],
                    sample["question"],
                    sample["answer"],
                )
                for sample in batch
            ]
            outputs = await asyncio.gather(*forward_tasks)

            # Check predictions (forward already returns extracted verdict)
            for sample, output in zip(batch, outputs, strict=True):
                prediction = str(output).strip().lower()
                if prediction == sample["label"]:
                    correct += 1
                total += 1

            progress.advance(val_task, len(batch))

    return correct / total if total > 0 else 0.0


# =============================================================================
# Training Loop
# =============================================================================


def create_progress() -> Progress:
    """Create a rich progress bar for training."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    )


def create_stats_table(
    epoch: int,
    num_epochs: int,
    batch_idx: int,
    num_batches: int,
    batch_correct: int,
    batch_total: int,
    epoch_correct: int,
    epoch_total: int,
) -> Table:
    """Create a table showing current training statistics."""
    table = Table(title="Training Statistics", show_header=True, header_style="bold")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    # Epoch info
    table.add_row("Epoch", f"{epoch + 1}/{num_epochs}")
    table.add_row("Batch", f"{batch_idx + 1}/{num_batches}")

    # Batch accuracy
    batch_acc = batch_correct / batch_total if batch_total > 0 else 0
    batch_style = (
        "green" if batch_acc >= 0.7 else "yellow" if batch_acc >= 0.5 else "red"
    )
    table.add_row(
        "Batch Accuracy",
        Text(f"{batch_correct}/{batch_total} ({batch_acc:.1%})", style=batch_style),
    )

    # Epoch accuracy
    epoch_acc = epoch_correct / epoch_total if epoch_total > 0 else 0
    epoch_style = (
        "green" if epoch_acc >= 0.7 else "yellow" if epoch_acc >= 0.5 else "red"
    )
    table.add_row(
        "Epoch Accuracy",
        Text(f"{epoch_correct}/{epoch_total} ({epoch_acc:.1%})", style=epoch_style),
    )

    return table


async def train_hallucination_detector() -> None:
    """Train the hallucination detection system."""
    if not os.environ.get("OPENAI_API_KEY"):
        console.print("[bold red]Error: OPENAI_API_KEY environment variable not set")
        sys.exit(1)

    # Header
    console.print()
    console.print(
        Panel.fit(
            "[bold cyan]inf-engine: Hallucination Detection Training[/bold cyan]\n\n"
            "Training a hallucination detector on HaluBench dataset\n"
            "[dim]Batch size: 32 | Forward: gpt-4o-mini | Optimizer: gpt-5.2[/dim]",
            border_style="cyan",
        )
    )
    console.print()

    # -------------------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------------------
    resources = ResourceConfig(
        endpoints={
            # Forward pass model
            "detector": OpenAIEndpointConfig(model="gpt-4o-mini", rate_limit=300),
            # Optimizer models (using gpt-5.2 as specified)
            "optimizer/aggregator": OpenAIEndpointConfig(
                model="gpt-5.2", rate_limit=100
            ),
            "optimizer/updater": OpenAIEndpointConfig(model="gpt-5.2", rate_limit=100),
        }
    )

    # Load dataset with train/validation split
    train_batches, val_samples = load_halubench_data(batch_size=32, val_size=256)

    # Create detector and optimizer
    detector = HallucinationDetector()
    optimizer = SFAOptimizer(detector.parameters(), conservatism=0.3)
    optimizer.bind(resources)

    # Show initial state
    console.print(
        Panel(
            str(detector.system_prompt.value),
            title="[bold]Initial System Prompt[/bold]",
            border_style="blue",
            padding=(1, 2),
        )
    )
    console.print(
        Panel(
            str(detector.user_prompt_template.value),
            title="[bold]Initial User Prompt Template[/bold]",
            border_style="blue",
            padding=(1, 2),
        )
    )
    console.print(
        Panel(
            f"[cyan]{detector.extractor.pattern.value}[/cyan]",
            title="[bold]Initial Extraction Regex[/bold]",
            border_style="blue",
            padding=(0, 2),
        )
    )
    console.print()

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------
    trace_path = Path("./traces/hallucination_training.json")
    trace_path.parent.mkdir(parents=True, exist_ok=True)

    # Training parameters
    num_epochs = 3
    max_batches_per_epoch = 5  # Limit batches per epoch for demo
    train_accuracies: list[float] = []
    val_accuracies: list[float] = []

    async with ExecutionSettings(
        resources=resources,
        profile=True,
        profile_path=trace_path,
    ) as settings:
        # Run initial validation before training
        console.print("[bold yellow]Running initial validation...[/bold yellow]")
        detector.eval()
        initial_val_accuracy = await run_validation(
            detector, val_samples, batch_size=32
        )
        val_accuracies.append(initial_val_accuracy)

        val_style = (
            "green"
            if initial_val_accuracy >= 0.7
            else "yellow"
            if initial_val_accuracy >= 0.5
            else "red"
        )
        console.print(
            Panel(
                f"[bold]Initial Val Accuracy: [{val_style}]{initial_val_accuracy:.1%}[/{val_style}][/bold]  "
                f"({int(initial_val_accuracy * len(val_samples))}/{len(val_samples)})",
                title="[bold]Baseline (Before Training)[/bold]",
                border_style="blue",
            )
        )
        console.print()

        # Enable training mode
        detector.train()

        for epoch in range(num_epochs):
            console.rule(f"[bold cyan]Epoch {epoch + 1}/{num_epochs}[/bold cyan]")

            epoch_correct = 0
            epoch_total = 0

            # Create progress bar for batches
            progress = create_progress()

            with progress:
                batch_task = progress.add_task(
                    f"Epoch {epoch + 1}",
                    total=min(len(train_batches), max_batches_per_epoch),
                )

                for batch_idx, batch in enumerate(
                    train_batches[:max_batches_per_epoch]
                ):
                    # Display current parameters at start of each batch
                    console.print()
                    console.print(
                        Panel(
                            str(detector.system_prompt.value),
                            title=f"[bold]Current System Prompt[/bold] [dim](Batch {batch_idx + 1})[/dim]",
                            border_style="magenta",
                            padding=(1, 2),
                        )
                    )
                    console.print(
                        Panel(
                            str(detector.user_prompt_template.value),
                            title="[bold]Current User Prompt Template[/bold]",
                            border_style="magenta",
                            padding=(1, 2),
                        )
                    )
                    console.print(
                        Panel(
                            f"[cyan]{detector.extractor.pattern.value}[/cyan]",
                            title="[bold]Current Extraction Regex[/bold]",
                            border_style="magenta",
                            padding=(0, 2),
                        )
                    )

                    optimizer.zero_feedback()

                    # ---------------------------------------------------------
                    # Forward pass: run all samples in batch in parallel
                    # ---------------------------------------------------------
                    progress.update(
                        batch_task, description=f"Epoch {epoch + 1} - Forward pass"
                    )
                    forward_tasks = [
                        detector(
                            sample["context"],
                            sample["question"],
                            sample["answer"],
                        )
                        for sample in batch
                    ]
                    outputs = await asyncio.gather(*forward_tasks)

                    # ---------------------------------------------------------
                    # Compute loss for each sample
                    # ---------------------------------------------------------
                    progress.update(
                        batch_task, description=f"Epoch {epoch + 1} - Computing loss"
                    )
                    batch_correct = 0
                    all_feedback = []

                    for sample, output in zip(batch, outputs, strict=True):
                        # Create verifier for this sample's ground truth
                        verifier = make_accuracy_verifier(sample["label"])
                        loss_fn = VerifierLoss(verifier)

                        # Compute feedback
                        feedback = await loss_fn(output)
                        all_feedback.append(feedback)

                        # Track accuracy
                        if feedback.score == 1.0:
                            batch_correct += 1

                    epoch_correct += batch_correct
                    epoch_total += len(batch)

                    # Show stats table
                    console.print()
                    console.print(
                        create_stats_table(
                            epoch=epoch,
                            num_epochs=num_epochs,
                            batch_idx=batch_idx,
                            num_batches=min(len(train_batches), max_batches_per_epoch),
                            batch_correct=batch_correct,
                            batch_total=len(batch),
                            epoch_correct=epoch_correct,
                            epoch_total=epoch_total,
                        )
                    )

                    # ---------------------------------------------------------
                    # Backward pass for each feedback
                    # ---------------------------------------------------------
                    progress.update(
                        batch_task, description=f"Epoch {epoch + 1} - Backward pass"
                    )
                    for feedback in all_feedback:
                        await feedback.backward(optimizer=optimizer)

                    # ---------------------------------------------------------
                    # Update parameters after each batch
                    # ---------------------------------------------------------
                    progress.update(
                        batch_task,
                        description=f"Epoch {epoch + 1} - Updating parameters",
                    )
                    updates = await optimizer.step()
                    if updates:
                        console.print(
                            f"[green]Updated {len(updates)} parameter(s)[/green]"
                        )

                    progress.advance(batch_task)

            # Epoch summary - training accuracy
            epoch_accuracy = epoch_correct / epoch_total if epoch_total > 0 else 0
            train_accuracies.append(epoch_accuracy)

            train_style = (
                "green"
                if epoch_accuracy >= 0.7
                else "yellow"
                if epoch_accuracy >= 0.5
                else "red"
            )

            # Run validation
            console.print()
            console.print("[bold yellow]Running validation...[/bold yellow]")

            # Temporarily switch to eval mode for validation
            detector.eval()
            val_accuracy = await run_validation(detector, val_samples, batch_size=32)
            detector.train()  # Switch back to training mode

            val_accuracies.append(val_accuracy)

            val_style = (
                "green"
                if val_accuracy >= 0.7
                else "yellow"
                if val_accuracy >= 0.5
                else "red"
            )

            # Show epoch results
            console.print()
            console.print(
                Panel(
                    f"[bold]Train Accuracy: [{train_style}]{epoch_accuracy:.1%}[/{train_style}][/bold]  "
                    f"({epoch_correct}/{epoch_total})\n"
                    f"[bold]Val Accuracy:   [{val_style}]{val_accuracy:.1%}[/{val_style}][/bold]  "
                    f"({int(val_accuracy * len(val_samples))}/{len(val_samples)})",
                    title=f"[bold]Epoch {epoch + 1} Complete[/bold]",
                    border_style="green",
                )
            )

        # Switch to eval mode
        detector.eval()

        # Get profiler statistics
        profiler = settings.get_profiler()

    # -------------------------------------------------------------------------
    # Results
    # -------------------------------------------------------------------------
    console.print()
    console.rule("[bold cyan]Training Progress[/bold cyan]")

    # Progress table with train and val accuracies
    progress_table = Table(show_header=True, header_style="bold")
    progress_table.add_column("Epoch", style="cyan", justify="center")
    progress_table.add_column("Train Acc", justify="center")
    progress_table.add_column("Val Acc", justify="center")

    # First row: initial validation (no training yet)
    initial_val_style = (
        "green"
        if val_accuracies[0] >= 0.7
        else "yellow"
        if val_accuracies[0] >= 0.5
        else "red"
    )
    progress_table.add_row(
        "Initial",
        Text("-", style="dim"),
        Text(f"{val_accuracies[0]:.1%}", style=initial_val_style),
    )

    # Subsequent rows: each epoch with train and val accuracies
    for i, (train_acc, val_acc) in enumerate(
        zip(train_accuracies, val_accuracies[1:], strict=True)
    ):
        train_style = (
            "green" if train_acc >= 0.7 else "yellow" if train_acc >= 0.5 else "red"
        )
        val_style = "green" if val_acc >= 0.7 else "yellow" if val_acc >= 0.5 else "red"
        progress_table.add_row(
            str(i + 1),
            Text(f"{train_acc:.1%}", style=train_style),
            Text(f"{val_acc:.1%}", style=val_style),
        )

    console.print(progress_table)

    # Compare initial validation to final validation
    if len(val_accuracies) >= 2:
        val_improvement = val_accuracies[-1] - val_accuracies[0]
        val_sign = "+" if val_improvement >= 0 else ""
        val_imp_style = (
            "green" if val_improvement > 0 else "red" if val_improvement < 0 else "dim"
        )
        console.print(
            f"\n[bold]Val Improvement (Initial → Final):[/bold] "
            f"[{val_imp_style}]{val_sign}{val_improvement:.1%}[/{val_imp_style}]"
        )

    if len(train_accuracies) >= 2:
        train_improvement = train_accuracies[-1] - train_accuracies[0]
        train_sign = "+" if train_improvement >= 0 else ""
        train_imp_style = (
            "green"
            if train_improvement > 0
            else "red"
            if train_improvement < 0
            else "dim"
        )
        console.print(
            f"[bold]Train Improvement (Epoch 1 → Final):[/bold] "
            f"[{train_imp_style}]{train_sign}{train_improvement:.1%}[/{train_imp_style}]"
        )

    # Final parameters
    console.print()
    console.rule("[bold cyan]Final Parameters[/bold cyan]")
    console.print(
        Panel(
            str(detector.system_prompt.value),
            title="[bold green]Optimized System Prompt[/bold green]",
            border_style="green",
            padding=(1, 2),
        )
    )
    console.print(
        Panel(
            str(detector.user_prompt_template.value),
            title="[bold green]Optimized User Prompt Template[/bold green]",
            border_style="green",
            padding=(1, 2),
        )
    )
    console.print(
        Panel(
            f"[cyan]{detector.extractor.pattern.value}[/cyan]",
            title="[bold green]Optimized Extraction Regex[/bold green]",
            border_style="green",
            padding=(0, 2),
        )
    )

    # -------------------------------------------------------------------------
    # Profiling Statistics
    # -------------------------------------------------------------------------
    if profiler:
        stats = profiler.get_statistics()
        console.print()
        console.rule("[bold cyan]Profiling Statistics[/bold cyan]")

        stats_table = Table(show_header=True, header_style="bold")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="green")

        stats_table.add_row("Total tasks", str(stats.total_tasks))
        stats_table.add_row("Completed", str(stats.completed_tasks))
        stats_table.add_row("Failed", str(stats.failed_tasks))
        stats_table.add_row("Total duration", f"{stats.total_duration_ms:.1f}ms")
        stats_table.add_row("Avg task time", f"{stats.avg_duration_ms:.1f}ms")

        console.print(stats_table)
        console.print(f"\n[dim]Trace exported to: {trace_path}[/dim]")

    console.print()
    console.print(
        Panel.fit(
            "[bold green]Training complete![/bold green]",
            border_style="green",
        )
    )


if __name__ == "__main__":
    asyncio.run(train_hallucination_detector())
