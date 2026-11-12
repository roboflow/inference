import assert from "node:assert/strict";
import fs from "node:fs";
import { Evaluator, Lexer, Parser, data } from "@actions/expressions";
import { parse } from "yaml";

const workflow = parse(fs.readFileSync("/work/.github/workflows/publish.pypi.yml", "utf8"));
const results = ["success", "failure", "cancelled", "skipped"];
const modes = ["sdk", "runtime", "all", "dependencies"];
let checks = 0;
let scenarios = 0;
function equal(actual, expected, label) {
  assert.deepEqual(actual, expected, label);
  checks++;
}
function convert(value) {
  if (value === null || value === undefined) return new data.Null();
  if (typeof value === "boolean") return new data.BooleanData(value);
  if (typeof value === "string") return new data.StringData(value);
  return new data.Dictionary(...Object.entries(value).map(([key, item]) => ({ key, value: convert(item) })));
}
function evaluate(source, context, status = null) {
  let expression = source?.trim() || "success()";
  if (expression.startsWith("${{")) expression = expression.slice(3, -2).trim();
  // The runner adds success() to conditions without an explicit status function.
  if (status && !/\b(success|failure|cancelled|always)\s*\(/.test(expression)) {
    expression = `success() && (${expression})`;
  }
  const functions = new Map(Object.entries({
    success: status?.success ?? true,
    failure: status?.failure ?? false,
    cancelled: status?.cancelled ?? false,
    always: true,
  }).map(([name, value]) => [name, { name, minArgs: 0, maxArgs: 0, call: () => new data.BooleanData(value) }]));
  const tokens = new Lexer(expression).lex().tokens;
  const ast = new Parser(tokens, ["github", "inputs", "needs", "steps", "env"], [...functions.values()]).parse();
  return new Evaluator(ast, convert(context), functions).evaluate().value;
}
function context(packages = "all", publish = true, event = "workflow_dispatch", repository = "roboflow/inference") {
  return {
    github: { repository, event_name: event, event: { release: { tag_name: "v1.2.3" } } },
    inputs: { packages, publish, expected_version: "1.2.4rc1" },
    needs: {}, steps: {}, env: {},
  };
}
function jobEnabled(job, ctx, cancelled = false) {
  const dependencies = [].concat(job.needs || []).map(name => ctx.needs[name]?.result);
  return evaluate(job.if, ctx, {
    success: !cancelled && dependencies.every(result => result === "success"),
    failure: dependencies.includes("failure"), cancelled,
  });
}
function stepEnabled(step, ctx, previous = "success", cancelled = false) {
  return evaluate(step.if, ctx, {
    success: previous !== "failure" && previous !== "cancelled" && !cancelled,
    failure: previous === "failure", cancelled,
  });
}
function publicationSteps(job, command) {
  const uploads = job.steps.filter(step => step.uses?.startsWith("pypa/gh-action-pypi-publish@"));
  const gates = job.steps.filter(step => step.run?.includes(`release.sh ${command} `));
  equal(uploads.length, 1, `${command}: exactly one upload`);
  equal(gates.length, 1, `${command}: mandatory verification step`);
  return { upload: uploads[0], gate: gates[0] };
}
function runSteps(job, ctx, outcomes) {
  ctx.steps = {};
  let result = "success";
  const ran = [];
  for (const step of job.steps) {
    const enabled = stepEnabled(step, ctx, result, result === "cancelled");
    const outcome = enabled ? outcomes.get(step) || "success" : "skipped";
    const tolerated = step["continue-on-error"] === true;
    const conclusion = outcome === "failure" && tolerated ? "success" : outcome;
    if (step.id) ctx.steps[step.id] = { outcome, conclusion, outputs: step.id === "prepare" && outcome === "success" ? { upload: "true" } : {} };
    if (enabled) ran.push(step);
    if (conclusion === "failure" || conclusion === "cancelled") result = conclusion;
  }
  return { result, ran };
}
function check(wf) {
  const jobs = wf.jobs;
  equal(wf.on.workflow_dispatch.inputs.packages.options, modes, "supported package modes");
  equal(wf.on.workflow_dispatch.inputs.publish.type, "boolean", "typed publication input");
  equal(wf.on.workflow_dispatch.inputs.publish.default, false, "build-only by default");
  equal(jobs.publish_sdk["timeout-minutes"], 30, "SDK preparation and verification budget");
  equal(jobs.publish_sdk.needs, "build_sdk", "SDK build ownership");
  equal(jobs.prepare_dependencies.needs, ["build_runtime", "publish_sdk"], "coordinator dependency ownership");
  equal(jobs.publish_runtime.needs, ["build_runtime", "publish_sdk", "prepare_dependencies"], "runtime dependency ownership");

  for (const packages of modes) for (const publish of [false, true]) {
    for (const event of ["workflow_dispatch", "release"]) for (const repository of ["roboflow/inference", "fork/inference"]) {
      const ctx = context(packages, publish, event, repository);
      const release = event === "release";
      const sdk = release || packages !== "runtime";
      const runtime = release || packages !== "sdk";
      const publishing = repository === "roboflow/inference" && (release || publish);
      const expected = {
        build_sdk: sdk,
        publish_sdk: sdk && publishing,
        build_runtime: runtime,
        prepare_dependencies: runtime && repository === "roboflow/inference",
        publish_runtime: runtime && publishing && (release || packages !== "dependencies"),
      };
      for (const name of ["build_sdk", "publish_sdk", "build_runtime", "prepare_dependencies", "publish_runtime"]) {
        const enabled = jobEnabled(jobs[name], ctx);
        equal(enabled, expected[name], `${packages}/${publish}/${event}/${repository}: ${name}`);
        ctx.needs[name] = { result: enabled ? "success" : "skipped" };
      }
      equal(evaluate(wf.env.PUBLISH, ctx), release || publish, "PUBLISH boolean expression");
      equal(evaluate(wf.env.EXPECTED_VERSION, ctx), release ? "v1.2.3" : "1.2.4rc1", "release tag or manual expected version");
      scenarios++;

      for (const sdkResult of results) for (const runtimeResult of results) for (const cancelled of [false, true]) {
        ctx.needs = { build_runtime: { result: runtimeResult }, publish_sdk: { result: sdkResult } };
        const coordinatorSdkReady = sdkResult === "success" || (!release && (packages === "runtime" || !publish) && sdkResult === "skipped");
        equal(jobEnabled(jobs.prepare_dependencies, ctx, cancelled),
          !cancelled && repository === "roboflow/inference" && runtimeResult === "success" && coordinatorSdkReady,
          `coordinator dependencies: ${JSON.stringify({ packages, publish, event, repository, sdkResult, runtimeResult, cancelled })}`);
        const sdkReady = sdkResult === "success" || (!release && packages === "runtime" && sdkResult === "skipped");
        for (const dependencyResult of results) {
          ctx.needs.prepare_dependencies = { result: dependencyResult };
          equal(jobEnabled(jobs.publish_runtime, ctx, cancelled),
            !cancelled && publishing && (release || packages !== "dependencies") && runtimeResult === "success" && dependencyResult === "success" && sdkReady,
            `runtime dependencies: ${JSON.stringify({ packages, publish, event, repository, sdkResult, runtimeResult, dependencyResult, cancelled })}`);
          scenarios++;
        }
        scenarios++;
      }
      for (const result of results) for (const cancelled of [false, true]) {
        ctx.needs = { build_sdk: { result } };
        equal(jobEnabled(jobs.publish_sdk, ctx, cancelled), publishing && result === "success" && !cancelled, "SDK build dependency and cancellation");
        scenarios++;
      }
    }
  }
  const ctx = context();
  ctx.inputs.expected_version = "";
  equal(evaluate(wf.env.EXPECTED_VERSION, ctx), "", "empty manual expected version reaches release validation");

  const sdk = publicationSteps(jobs.publish_sdk, "verify-sdk");
  const runtime = publicationSteps(jobs.publish_runtime, "gate-sdk");
  const prepare = jobs.publish_sdk.steps.find(step => step.id === "prepare");
  assert.ok(prepare?.run.includes("release.sh prepare-sdk --wait-seconds 600"), "SDK preparation remains bounded");
  equal(sdk.upload.id, "upload", "record SDK upload outcome");
  equal(sdk.upload["continue-on-error"], true, "reconcile a failed SDK upload");
  equal(sdk.gate["continue-on-error"] ?? false, false, "strict verification controls SDK job result");
  equal(runtime.upload["continue-on-error"] ?? false, false, "runtime upload errors fail");
  equal(runtime.gate["continue-on-error"] ?? false, false, "runtime gate errors fail");
  assert.ok(jobs.publish_sdk.steps.indexOf(prepare) < jobs.publish_sdk.steps.indexOf(sdk.upload));
  assert.ok(jobs.publish_sdk.steps.indexOf(sdk.upload) < jobs.publish_sdk.steps.indexOf(sdk.gate));
  assert.ok(jobs.publish_runtime.steps.indexOf(runtime.gate) < jobs.publish_runtime.steps.indexOf(runtime.upload));
  equal(sdk.upload.with["packages-dir"], "dist/sdk-upload/", "SDK upload ownership");
  equal(runtime.upload.with["packages-dir"], "dist/pypi/", "runtime upload ownership");
  equal(sdk.upload.with["skip-existing"] ?? false, false, "SDK recovery uses verification, not skip-existing");

  for (const upload of ["true", "false", "", undefined]) for (const previous of results) for (const cancelled of [false, true]) {
    ctx.steps = { prepare: { outputs: { upload }, outcome: previous, conclusion: previous } };
    equal(stepEnabled(sdk.upload, ctx, previous, cancelled), upload === "true" && previous !== "failure" && previous !== "cancelled" && !cancelled, "SDK upload output and prior status");
    equal(stepEnabled(sdk.gate, ctx, previous, cancelled), previous !== "failure" && previous !== "cancelled" && !cancelled, "verification cannot be bypassed after successful preparation");
    equal(stepEnabled(runtime.gate, ctx, previous, cancelled), previous !== "failure" && previous !== "cancelled" && !cancelled, "runtime gate honors prior failures");
    equal(stepEnabled(runtime.upload, ctx, previous, cancelled), previous !== "failure" && previous !== "cancelled" && !cancelled, "runtime upload honors failed gate");
    scenarios++;
  }

  for (const uploadResult of ["success", "failure"]) for (const verificationResult of ["success", "failure"]) {
    const execution = runSteps(jobs.publish_sdk, ctx, new Map([[sdk.upload, uploadResult], [sdk.gate, verificationResult]]));
    equal(execution.ran.includes(sdk.gate), true, "verification runs after tolerated upload failure");
    equal(ctx.steps.upload.outcome, uploadResult, "upload outcome preserved");
    equal(ctx.steps.upload.conclusion, "success", "tolerated upload conclusion permits reconciliation");
    equal(execution.result, verificationResult, "strict verification determines SDK job result");
    ctx.needs = { build_runtime: { result: "success" }, publish_sdk: { result: execution.result }, prepare_dependencies: { result: "success" } };
    equal(jobEnabled(jobs.prepare_dependencies, ctx), verificationResult === "success", "final SDK verification controls dependency coordination");
    equal(jobEnabled(jobs.publish_runtime, ctx), verificationResult === "success", "final SDK verification controls runtime publication");
    scenarios++;
  }
  for (const preparation of ["failure", "cancelled", "skipped"]) {
    const execution = runSteps(jobs.publish_sdk, ctx, new Map([[prepare, preparation], [sdk.gate, "failure"]]));
    equal(execution.ran.includes(sdk.upload), false, "failed or skipped preparation cannot upload");
    equal(execution.result, preparation === "cancelled" ? "cancelled" : "failure", "missing preparation cannot produce a verified SDK");
    scenarios++;
  }
  for (const gateResult of results) {
    const execution = runSteps(jobs.publish_runtime, ctx, new Map([[runtime.gate, gateResult]]));
    equal(execution.ran.includes(runtime.upload), !["failure", "cancelled"].includes(gateResult), "runtime upload follows evaluated gate status");
    scenarios++;
  }
  for (const name of Object.keys(jobs)) {
    if (name.startsWith("publish_")) {
      equal(jobs[name].environment, undefined, "existing OIDC publisher has no environment");
      equal(jobs[name].permissions, { contents: "read", "id-token": "write" }, "publisher OIDC permissions");
    }
    for (const step of jobs[name].steps) {
      if (step.env?.PUBLISH !== undefined) equal(step.env.PUBLISH, wf.env.PUBLISH, "no PUBLISH override");
      if (step.env?.EXPECTED_VERSION !== undefined) equal(step.env.EXPECTED_VERSION, wf.env.EXPECTED_VERSION, "no expected-version override");
    }
    equal(jobs[name].env?.PUBLISH, undefined, "inherit PUBLISH at job scope");
    equal(jobs[name].env?.EXPECTED_VERSION, undefined, "inherit EXPECTED_VERSION at job scope");
  }
  for (const publish of [false, true]) {
    ctx.env = { PUBLISH: String(publish) };
    for (const name of ["build_sdk", "build_runtime"]) {
      const summary = jobs[name].steps.find(step => step.run?.includes("Nothing was published"));
      assert.ok(summary, "build-only result is visible");
      equal(stepEnabled(summary, ctx), !publish, "build-only summary uses explicit env boolean comparison");
      equal(stepEnabled(summary, ctx, "failure"), false, "failed builds do not claim success");
    }
  }
}

check(workflow);
console.log(`Workflow conditions: ${checks} checks passed across ${scenarios} scenarios`);
const mutations = [
  ["ignore SDK failure", wf => { wf.jobs.publish_runtime.if = wf.jobs.publish_runtime.if.replace("needs.publish_sdk.result == 'success'", "true"); }],
  ["always after cancellation", wf => { wf.jobs.publish_runtime.if = wf.jobs.publish_runtime.if.replace("!cancelled()", "always()"); }],
  ["build any mode", wf => { wf.jobs.build_runtime.if = "${{ true }}"; }],
  ["drop repository guard", wf => { wf.jobs.publish_runtime.if = wf.jobs.publish_runtime.if.replace("github.repository == 'roboflow/inference' &&", ""); }],
  ["unconditional SDK upload", wf => { delete wf.jobs.publish_sdk.steps.find(step => step.id === "upload").if; }],
  ["PUBLISH hardcoded true", wf => { wf.env.PUBLISH = "${{ true }}"; }],
  ["EXPECTED_VERSION loses release tag", wf => { wf.env.EXPECTED_VERSION = "${{ inputs.expected_version }}"; }],
  ["bare string PUBLISH in summary", wf => { for (const name of ["build_sdk", "build_runtime"]) wf.jobs[name].steps.find(step => step.run?.includes("Nothing was published")).if = "${{ env.PUBLISH }}"; }],
  ["delete SDK verification", wf => { wf.jobs.publish_sdk.steps = wf.jobs.publish_sdk.steps.filter(step => !step.run?.includes("release.sh verify-sdk ")); }],
];
for (const [name, mutate] of mutations) {
  const faulty = structuredClone(workflow);
  mutate(faulty);
  assert.notDeepEqual(faulty, workflow, `${name}: mutation applied`);
  assert.throws(() => check(faulty), assert.AssertionError, `${name}: rejected`);
  console.log(`Mutation rejected: ${name}`);
}
console.log(`Workflow mutations: ${mutations.length}/${mutations.length} rejected`);
