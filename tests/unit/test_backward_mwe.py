"""MWE demonstrating feedback propagation gaps in backward traversal."""

import pytest

from plait.graph import GraphNode, InferenceGraph, NodeRef
from plait.module import Module
from plait.optimization.backward import BackwardContext, BackwardResult, _propagate_backward
from plait.optimization.feedback import Feedback, FeedbackType
from plait.optimization.optimizer import Optimizer
from plait.optimization.record import ForwardRecord
from plait.parameter import Parameter


class UpstreamRecorder(Module):
    """Records the feedback content it receives during backward."""

    def __init__(self) -> None:
        super().__init__()
        self.backward_contents: list[str] = []
        self.backward_feedbacks: list[Feedback] = []

    def forward(self, x: str) -> str:
        return f"up_{x}"

    async def backward(
        self,
        feedback: Feedback,
        ctx: BackwardContext,
    ) -> BackwardResult:
        self.backward_contents.append(feedback.content)
        self.backward_feedbacks.append(feedback)
        return await super().backward(feedback, ctx)


class DownstreamTransformer(Module):
    """Transforms feedback before propagating to its input."""

    def forward(self, x: str) -> str:
        return f"down_{x}"

    async def backward(
        self,
        feedback: Feedback,
        ctx: BackwardContext,
    ) -> BackwardResult:
        result = BackwardResult()
        result.input_feedback["0"] = Feedback(
            content="transformed-by-downstream",
            score=feedback.score,
            feedback_type=feedback.feedback_type,
        )
        return result


class ParamDownstreamTransformer(Module):
    """Transforms feedback with a configurable tag."""

    def __init__(self, tag: str, score: float) -> None:
        super().__init__()
        self.tag = tag
        self.score = score

    def forward(self, x: str) -> str:
        return f"{self.tag}_{x}"

    async def backward(
        self,
        feedback: Feedback,
        ctx: BackwardContext,
    ) -> BackwardResult:
        result = BackwardResult()
        result.input_feedback["0"] = Feedback(
            content=f"transformed-{self.tag}",
            score=self.score,
            feedback_type=feedback.feedback_type,
        )
        return result


class BranchTransformerKeepScore(Module):
    """Transforms feedback content while preserving incoming score."""

    def __init__(self, tag: str) -> None:
        super().__init__()
        self.tag = tag
        self.received_contents: list[str] = []

    def forward(self, x: str) -> str:
        return f"{self.tag}_{x}"

    async def backward(
        self,
        feedback: Feedback,
        ctx: BackwardContext,
    ) -> BackwardResult:
        self.received_contents.append(feedback.content)
        result = BackwardResult()
        result.input_feedback["0"] = Feedback(
            content=f"transformed-{self.tag}",
            score=feedback.score,
            feedback_type=feedback.feedback_type,
        )
        return result


class JoinFanOut(Module):
    """Joins two inputs and fans feedback out to both branches."""

    def __init__(self) -> None:
        super().__init__()
        self.join_tag = Parameter(
            value="join",
            description="Unused join tag (should not receive feedback)",
            requires_grad=True,
        )

    def forward(self, left: str, right: str) -> str:
        return f"join_{left}_{right}"

    async def backward(
        self,
        feedback: Feedback,
        ctx: BackwardContext,
    ) -> BackwardResult:
        result = BackwardResult()
        result.input_feedback["0"] = Feedback(
            content="to-left",
            score=0.2,
            feedback_type=feedback.feedback_type,
        )
        result.input_feedback["1"] = Feedback(
            content="to-right",
            score=0.8,
            feedback_type=feedback.feedback_type,
        )
        return result


class UpstreamWithParam(Module):
    """Upstream module with a learnable parameter."""

    def __init__(self) -> None:
        super().__init__()
        self.instructions = Parameter(
            value="base",
            description="Instructions for response generation",
            requires_grad=True,
        )
        self.backward_contents: list[str] = []

    def forward(self, x: str) -> str:
        return f"{self.instructions.value}_{x}"

    async def backward(
        self,
        feedback: Feedback,
        ctx: BackwardContext,
    ) -> BackwardResult:
        self.backward_contents.append(feedback.content)
        result = BackwardResult()
        result.parameter_feedback["instructions"] = f"update for {feedback.content}"
        for input_name in ctx.inputs:
            result.input_feedback[input_name] = feedback
        return result


class DummyOptimizer(Optimizer):
    """Deterministic optimizer that applies accumulated feedback directly."""

    async def step(self) -> dict[str, str]:
        updates: dict[str, str] = {}
        for param in self.params:
            if param._feedback_buffer:
                new_value = f"{param.value} | {param._feedback_buffer[-1]}"
                param.apply_update(new_value)
                updates[self._param_key(param)] = new_value
        return updates


@pytest.mark.asyncio
async def test_mwe_transformed_feedback_reaches_upstream() -> None:
    """Upstream node should receive downstream-transformed feedback.

    Diagram:
        input:x ("hello")
          |
          v
        UpstreamRecorder
          |
          v
        DownstreamTransformer (output)

    Backward:
        loss-feedback -> DownstreamTransformer.backward
        transformed-by-downstream -> UpstreamRecorder.backward
    """
    upstream = UpstreamRecorder()
    downstream = DownstreamTransformer()

    # Graph: input -> upstream -> downstream (output)
    input_node = GraphNode(
        id="input:x",
        module=None,
        args=(),
        kwargs={},
        dependencies=[],
    )
    upstream_node = GraphNode(
        id="Upstream",
        module=upstream,
        args=(NodeRef("input:x"),),
        kwargs={},
        dependencies=["input:x"],
    )
    downstream_node = GraphNode(
        id="Downstream",
        module=downstream,
        args=(NodeRef("Upstream"),),
        kwargs={},
        dependencies=["Upstream"],
    )

    graph = InferenceGraph(
        nodes={
            "input:x": input_node,
            "Upstream": upstream_node,
            "Downstream": downstream_node,
        },
        input_ids=["input:x"],
        output_ids=["Downstream"],
    )

    record = ForwardRecord(
        graph=graph,
        node_inputs={
            "Upstream": {"0": "hello"},
            "Downstream": {"0": "up_hello"},
        },
        node_outputs={
            "input:x": "hello",
            "Upstream": "up_hello",
            "Downstream": "down_up_hello",
        },
        module_map={
            "Upstream": upstream,
            "Downstream": downstream,
        },
    )

    feedback = Feedback(content="loss-feedback", score=0.4)
    await _propagate_backward(feedback, record)

    # This is the expected behavior for a correct traversal.
    assert upstream.backward_contents == ["transformed-by-downstream"]
    assert upstream.backward_feedbacks[0].feedback_type == FeedbackType.HUMAN


@pytest.mark.asyncio
async def test_mwe_fan_in_feedback_accumulates() -> None:
    """Upstream node should receive combined feedback from multiple downstreams.

    Diagram:
        input:x ("hello")
          |
          v
        UpstreamRecorder
         /        \
        v          v
      DownA      DownB

    Backward:
        loss-feedback -> DownA.backward -> transformed-a
        loss-feedback -> DownB.backward -> transformed-b
        transformed-a + transformed-b -> UpstreamRecorder.backward (combined)
    """
    upstream = UpstreamRecorder()
    downstream_a = ParamDownstreamTransformer("a", score=0.2)
    downstream_b = ParamDownstreamTransformer("b", score=0.8)

    # Graph: input -> upstream -> [downstream_a, downstream_b]
    input_node = GraphNode(
        id="input:x",
        module=None,
        args=(),
        kwargs={},
        dependencies=[],
    )
    upstream_node = GraphNode(
        id="Upstream",
        module=upstream,
        args=(NodeRef("input:x"),),
        kwargs={},
        dependencies=["input:x"],
    )
    down_a_node = GraphNode(
        id="DownA",
        module=downstream_a,
        args=(NodeRef("Upstream"),),
        kwargs={},
        dependencies=["Upstream"],
    )
    down_b_node = GraphNode(
        id="DownB",
        module=downstream_b,
        args=(NodeRef("Upstream"),),
        kwargs={},
        dependencies=["Upstream"],
    )

    graph = InferenceGraph(
        nodes={
            "input:x": input_node,
            "Upstream": upstream_node,
            "DownA": down_a_node,
            "DownB": down_b_node,
        },
        input_ids=["input:x"],
        output_ids=["DownA", "DownB"],
    )

    record = ForwardRecord(
        graph=graph,
        node_inputs={
            "Upstream": {"0": "hello"},
            "DownA": {"0": "up_hello"},
            "DownB": {"0": "up_hello"},
        },
        node_outputs={
            "input:x": "hello",
            "Upstream": "up_hello",
            "DownA": "a_up_hello",
            "DownB": "b_up_hello",
        },
        module_map={
            "Upstream": upstream,
            "DownA": downstream_a,
            "DownB": downstream_b,
        },
    )

    feedback = Feedback(content="loss-feedback", score=0.4)
    await _propagate_backward(feedback, record)

    assert len(upstream.backward_contents) == 1
    assert len(upstream.backward_feedbacks) == 1
    assert upstream.backward_feedbacks[0].feedback_type == FeedbackType.COMPOSITE
    assert upstream.backward_feedbacks[0].score == pytest.approx(0.5)
    combined = upstream.backward_contents[0]
    assert "transformed-a" in combined
    assert "transformed-b" in combined


@pytest.mark.asyncio
async def test_mwe_backward_to_update_flow() -> None:
    """Full backward -> update flow uses transformed feedback for updates.

    Diagram:
        input:x ("hello")
          |
          v
        UpstreamWithParam (instructions="base")
          |
          v
        DownstreamTransformer (output)

    Backward:
        loss-feedback -> DownstreamTransformer.backward
        transformed-by-downstream -> UpstreamWithParam.backward
        parameter_feedback["instructions"] = "update for transformed-by-downstream"

    Step:
        instructions.value = "base | update for transformed-by-downstream"
    """
    upstream = UpstreamWithParam()
    downstream = DownstreamTransformer()

    # Graph: input -> upstream -> downstream (output)
    input_node = GraphNode(
        id="input:x",
        module=None,
        args=(),
        kwargs={},
        dependencies=[],
    )
    upstream_node = GraphNode(
        id="Upstream",
        module=upstream,
        args=(NodeRef("input:x"),),
        kwargs={},
        dependencies=["input:x"],
    )
    downstream_node = GraphNode(
        id="Downstream",
        module=downstream,
        args=(NodeRef("Upstream"),),
        kwargs={},
        dependencies=["Upstream"],
    )

    graph = InferenceGraph(
        nodes={
            "input:x": input_node,
            "Upstream": upstream_node,
            "Downstream": downstream_node,
        },
        input_ids=["input:x"],
        output_ids=["Downstream"],
    )

    record = ForwardRecord(
        graph=graph,
        node_inputs={
            "Upstream": {"0": "hello"},
            "Downstream": {"0": "base_hello"},
        },
        node_outputs={
            "input:x": "hello",
            "Upstream": "base_hello",
            "Downstream": "down_base_hello",
        },
        module_map={
            "Upstream": upstream,
            "Downstream": downstream,
        },
    )

    optimizer = DummyOptimizer(upstream.parameters())
    feedback = Feedback(content="loss-feedback", score=0.4)
    feedback._records = [record]

    await feedback.backward(optimizer=optimizer)
    updates = await optimizer.step()

    assert upstream.backward_contents == ["transformed-by-downstream"]
    updated_value = upstream.instructions.value
    assert "update for transformed-by-downstream" in updated_value
    assert updates[optimizer._param_key(upstream.instructions)] == updated_value
    assert upstream.instructions._feedback_buffer == []


@pytest.mark.asyncio
async def test_mwe_fan_out_then_fan_in() -> None:
    """Fan-out to middle branches then fan-in to a single upstream node.

    Diagram:
        input:x ("hello")
          |
          v
        UpstreamRecorder
         /        \\
        v          v
      BranchA    BranchB
         \\        /
          v      v
            Join (output)

    Backward:
        loss-feedback -> JoinFanOut.backward
        to-left -> BranchA.backward -> transformed-a
        to-right -> BranchB.backward -> transformed-b
        transformed-a + transformed-b -> UpstreamRecorder.backward (combined)
    """
    upstream = UpstreamRecorder()
    branch_a = BranchTransformerKeepScore("a")
    branch_b = BranchTransformerKeepScore("b")
    join = JoinFanOut()

    input_node = GraphNode(
        id="input:x",
        module=None,
        args=(),
        kwargs={},
        dependencies=[],
    )
    upstream_node = GraphNode(
        id="Upstream",
        module=upstream,
        args=(NodeRef("input:x"),),
        kwargs={},
        dependencies=["input:x"],
    )
    branch_a_node = GraphNode(
        id="BranchA",
        module=branch_a,
        args=(NodeRef("Upstream"),),
        kwargs={},
        dependencies=["Upstream"],
    )
    branch_b_node = GraphNode(
        id="BranchB",
        module=branch_b,
        args=(NodeRef("Upstream"),),
        kwargs={},
        dependencies=["Upstream"],
    )
    join_node = GraphNode(
        id="Join",
        module=join,
        args=(NodeRef("BranchA"), NodeRef("BranchB")),
        kwargs={},
        dependencies=["BranchA", "BranchB"],
    )

    graph = InferenceGraph(
        nodes={
            "input:x": input_node,
            "Upstream": upstream_node,
            "BranchA": branch_a_node,
            "BranchB": branch_b_node,
            "Join": join_node,
        },
        input_ids=["input:x"],
        output_ids=["Join"],
    )

    record = ForwardRecord(
        graph=graph,
        node_inputs={
            "Upstream": {"0": "hello"},
            "BranchA": {"0": "up_hello"},
            "BranchB": {"0": "up_hello"},
            "Join": {"0": "a_up_hello", "1": "b_up_hello"},
        },
        node_outputs={
            "input:x": "hello",
            "Upstream": "up_hello",
            "BranchA": "a_up_hello",
            "BranchB": "b_up_hello",
            "Join": "join_a_up_hello_b_up_hello",
        },
        module_map={
            "Upstream": upstream,
            "BranchA": branch_a,
            "BranchB": branch_b,
            "Join": join,
        },
    )

    feedback = Feedback(content="loss-feedback", score=0.4)
    await _propagate_backward(feedback, record)

    assert branch_a.received_contents == ["to-left"]
    assert branch_b.received_contents == ["to-right"]
    assert join.join_tag._feedback_buffer == []
    assert len(upstream.backward_contents) == 1
    combined = upstream.backward_contents[0]
    assert "transformed-a" in combined
    assert "transformed-b" in combined
    assert upstream.backward_feedbacks[0].feedback_type == FeedbackType.COMPOSITE
    assert upstream.backward_feedbacks[0].score == pytest.approx(0.5)
