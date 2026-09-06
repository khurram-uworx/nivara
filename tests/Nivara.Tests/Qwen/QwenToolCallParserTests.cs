using NivaraChat.Qwen;
using NUnit.Framework;

namespace Nivara.Tests.Qwen;

/// <summary>
/// Covers <see cref="QwenToolCallParser"/>: canonical <c>&lt;tool_call&gt;</c> parsing, tolerant
/// fallback for ragged model output, name canonicalization against the registered tools, and
/// argument-dictionary fidelity (the Phase B silent-failure fix — <c>FunctionCallContent</c> must
/// carry a correct dict, not a mangled string).
/// </summary>
[TestFixture]
public class QwenToolCallParserTests
{
    const string KnownTool = "getWeather";

    [Test]
    public void Parse_CanonicalToolCall_ReturnsNameAndArguments()
    {
        const string text = "I'll check that.<tool_call>\n{\"name\": \"getWeather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>";

        var calls = QwenToolCallParser.Parse(text, [KnownTool]);

        Assert.That(calls, Has.Count.EqualTo(1));
        Assert.That(calls[0].Name, Is.EqualTo("getWeather"));
        Assert.That(calls[0].CallId, Does.StartWith("call_"));
        Assert.That(calls[0].Arguments, Is.Not.Null);
        var city = (System.Text.Json.JsonElement)calls[0].Arguments!["city"]!;
        Assert.That(city.ToString(), Is.EqualTo("Paris"));
    }

    [Test]
    public void Parse_CompactJson_IsTolerantOfWhitespace()
    {
        const string text = "<tool_call>{\"name\":\"getWeather\",\"arguments\":{\"city\":\"Paris\"}}</tool_call>";

        var calls = QwenToolCallParser.Parse(text);

        Assert.That(calls, Has.Count.EqualTo(1));
        Assert.That(calls[0].Name, Is.EqualTo("getWeather"));
        Assert.That(calls[0].Arguments, Is.Not.Null);
        Assert.That(calls[0].Arguments!.ContainsKey("city"), Is.True);
    }

    [Test]
    public void Parse_MultipleToolCalls_ReturnsAll()
    {
        const string text =
            "<tool_call>\n{\"name\": \"getWeather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>" +
            "<tool_call>\n{\"name\": \"getWeather\", \"arguments\": {\"city\": \"New York\"}}\n</tool_call>";

        var calls = QwenToolCallParser.Parse(text, [KnownTool]);

        Assert.That(calls, Has.Count.EqualTo(2));
        var secondCity = (System.Text.Json.JsonElement)calls[1].Arguments!["city"]!;
        Assert.That(secondCity.ToString(), Is.EqualTo("New York"));
    }

    [Test]
    public void Parse_CaseInsensitiveName_CanonicalizesToRegisteredTool()
    {
        const string text = "<tool_call>{ \"name\": \"GetWeather\", \"arguments\": {\"city\": \"Paris\"} }</tool_call>";

        var calls = QwenToolCallParser.Parse(text, [KnownTool]);

        Assert.That(calls, Has.Count.EqualTo(1));
        Assert.That(calls[0].Name, Is.EqualTo("getWeather"));
    }

    [Test]
    public void Parse_UnknownName_StaysAsEmitted()
    {
        const string text = "<tool_call>{ \"name\": \"unknownTool\", \"arguments\": {\"x\": 1} }</tool_call>";

        var calls = QwenToolCallParser.Parse(text, [KnownTool]);

        Assert.That(calls, Has.Count.EqualTo(1));
        Assert.That(calls[0].Name, Is.EqualTo("unknownTool"));
    }

    [Test]
    public void Parse_NoToolCall_ReturnsEmpty()
    {
        var calls = QwenToolCallParser.Parse("That's a plain answer, no tools needed.");

        Assert.That(calls, Is.Empty);
    }

    [Test]
    public void Parse_MalformedInnerJson_FallsBackTolerantlyWithRawPayload()
    {
        // Unquoted keys fail JsonDocument.Parse; the tolerant fallback still extracts the name.
        const string text = "<tool_call>{name: \"getWeather\", arguments: {city: \"Paris\"}}</tool_call>";

        var calls = QwenToolCallParser.Parse(text, [KnownTool]);

        Assert.That(calls, Has.Count.EqualTo(1));
        Assert.That(calls[0].Name, Is.EqualTo("getWeather"));
        Assert.That(calls[0].Arguments, Is.Not.Null);
        Assert.That(calls[0].Arguments!.ContainsKey("__raw"), Is.True);
    }
}