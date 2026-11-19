import assert from "node:assert/strict";
import fs from "node:fs";
import { Evaluator, Lexer, Parser, data } from "@actions/expressions";
import { parse } from "yaml";

const workflow = parse(fs.readFileSync("/work/.github/workflows/publish.pypi.yml", "utf8"));
const upstreams = Object.fromEntries([
  ["inference-models", "inference_exp", "inference_models/dist"],
  ["roboflow-workflows", "roboflow_workflows", "dist"],
  ["streamvision", "streamvision", "dist"],
].map(([name, suffix, directory]) => [name, { directory, workflow: parse(fs.readFileSync(`/work/.github/workflows/publish.pypi.${suffix}.yml`, "utf8")) }]));
const results = ["success", "failure", "cancelled", "skipped"];
const modes = ["all", "sdk"];
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
    github: { repository, event_name: event, event: { release: { tag_name: "v1.2.3", prerelease: false } } },
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
  equal(wf["cache-mode"], "none", "release cannot read or poison Actions caches");
  for (const job of Object.values(jobs)) {
    equal(job["cache-mode"] ?? "none", "none", "jobs cannot restore cache access");
  }
  const policy = JSON.parse(fs.readFileSync("/work/.release/release-policy.json", "utf8"));
  const suites = policy.rc_tests.map(name => "rc_" + name.replace(/\.yml$/, "").replaceAll(".", "_"));
  equal(wf.on.release.types, ["published"], "only published official releases trigger stable");
  equal(wf.on.workflow_dispatch.inputs.packages.options, modes, "supported package modes");
  equal(wf.on.workflow_dispatch.inputs.publish.default, false, "build-only default");
  equal(wf.concurrency["cancel-in-progress"], false, "publishing runs cannot cancel each other");
  for (const [name, expected] of Object.entries({
    build_sdk: ["prepare"], publish_sdk: ["prepare", "build_sdk"],
    build_models: ["prepare", "publish_sdk"], publish_models: ["prepare", "build_models"],
    build_workflows: ["prepare", "publish_models"], publish_workflows: ["prepare", "build_workflows"],
    build_streamvision: ["prepare", "publish_workflows"], publish_streamvision: ["prepare", "build_streamvision"],
    commit_locks: ["prepare", "publish_streamvision"], build_runtime: ["prepare", "commit_locks"],
    smoke: ["prepare", "commit_locks", "build_runtime"],
    validation: ["prepare", "commit_locks", "smoke", ...suites],
    prepare_dependencies: ["prepare", "commit_locks", "build_runtime", "validation"],
    publish_runtime: ["prepare", "commit_locks", "build_runtime", "prepare_dependencies"],
    docker: ["prepare", "commit_locks", "prepare_dependencies", "publish_runtime", "build_sdk", "build_runtime"],
    complete: ["prepare", "commit_locks", "prepare_dependencies", "publish_runtime", "docker"],
  })) equal([].concat(jobs[name].needs), expected, `${name}: mandatory dependencies`);
  for (const name of ["publish_models", "publish_workflows", "publish_streamvision"]) {
    equal(jobs[name].env.EXPECTED_VERSION, "", "independent project versions must not inherit the runtime version");
  }
  equal(jobs.commit_locks.permissions, { contents: "write" }, "only commit job writes source");
  equal(jobs.commit_locks.steps.find(step => step.id === "commit").env.GH_TOKEN, "${{ github.token }}", "GITHUB_TOKEN suppresses new CI runs");
  assert.ok(jobs.prepare.steps.some(step => step.run?.includes("flow.sh prepare")), "stable RC evidence gate precedes any publication");
  assert.ok(jobs.commit_locks.steps.some(step => step.run?.includes("flow.sh equivalence")), "prepared lock equivalence gate");
  for (const [name, job] of Object.entries(jobs)) {
    if (name !== "notify-svis") equal(job.permissions?.actions === "write", false, "no child run dispatch permission");
    for (const step of job.steps || []) {
      assert.ok(!/dependencies.sh coordinate|workflow run|actions\/workflows.*dispatches/.test(step.run || ""), "one workflow run");
    }
  }
  // The producer artifact, dependency edge and extraction root must all agree.
  for (const job of Object.values(jobs)) {
    for (const step of job.steps || []) {
      if (!step.uses?.startsWith("actions/download-artifact@") || !step.with?.name) continue;
      const match = step.with.name.match(/^\$\{\{ needs\.([a-z_]+)\.outputs\.([a-z_]+) \}\}$/);
      assert.ok(match, "artifact name comes from its successful producer output");
      const [, producer, output] = match;
      assert.ok([].concat(job.needs || []).includes(producer), "artifact producer is a dependency");
      const producerName = jobs[producer].outputs[output];
      assert.ok(jobs[producer].steps.some(s => s.uses?.startsWith("actions/upload-artifact@") && s.with.name === producerName), "download uses the exact retained producer artifact");
    }
  }
  const download = (job, producer) => jobs[job].steps.find(s => s.uses?.startsWith("actions/download-artifact@") && s.with.name.includes(`needs.${producer}.outputs`));
  equal(download("publish_sdk", "build_sdk").with.path, "dist", "SDK artifact preserves sdk subdirectory under dist");
  equal(download("build_models", "publish_sdk").with.path, "sdk-evidence", "SDK receipt uses multi-root artifact extraction");
  assert.ok(jobs.build_models.steps.some(s => s.run === "cp -R sdk-evidence/release-state/smoke release-state/smoke"), "use actual SDK inventory location");
  equal(download("complete", "publish_runtime").with.path, "release-state", "single runtime receipt extracts directly into release state");
  for (const [index, name] of suites.entries()) {
    equal(jobs[name].uses, `./.github/workflows/${policy.rc_tests[index]}`, "full suite called in same run");
    equal(jobs[name].with.release_source, "${{ needs.commit_locks.outputs.source_sha }}", "test generated lock commit");
  }
  for (const channel of ["preview", "sdk", "rc", "stable"]) for (const packages of modes) {
    const ctx = context(packages, channel !== "preview", channel === "stable" ? "release" : "workflow_dispatch");
    for (const name of Object.keys(jobs)) ctx.needs[name] = { result: "success", outputs: {} };
    ctx.needs.prepare.outputs.channel = channel;
    equal(jobEnabled(jobs.publish_sdk, ctx), channel !== "preview", "SDK publish gate");
    for (const name of ["build_models", "publish_models", "build_workflows", "publish_workflows", "build_streamvision", "publish_streamvision"]) {
      equal(jobEnabled(jobs[name], ctx), ["rc", "stable"].includes(channel), `${name}: release channel only`);
    }
    for (const name of suites) equal(jobEnabled(jobs[name], ctx), channel === "rc", "full tests run only for RC");
    equal(jobEnabled(jobs.preview_runtime, ctx), channel === "preview" && packages !== "sdk", "preview is separate build-only path");
    equal(evaluate(wf.env.PUBLISH, ctx), channel !== "preview", "explicit publish boolean");
    equal(evaluate(wf.env.EXPECTED_VERSION, ctx), channel === "stable" ? "v1.2.3" : "1.2.4rc1", "tag or input supplies expected version");
    for (const result of results) {
      ctx.needs.commit_locks.result = result;
      equal(jobEnabled(jobs.validation, ctx), result === "success", "validation handles skipped RC jobs on stable");
      equal(jobEnabled(jobs.validation, ctx, true), false, "validation cannot outlive cancellation");
    }
    scenarios++;
  }
  for (const prerelease of [false, true]) for (const repo of ["roboflow/inference", "fork/inference"]) {
    const ctx = context("all", true, "release", repo);
    ctx.github.event.release.prerelease = prerelease;
    equal(jobEnabled(jobs.prepare, ctx), !prerelease && repo === "roboflow/inference", "only official repository stable release");
    scenarios++;
  }
  for (const name of ["publish_sdk", "publish_models", "publish_workflows", "publish_streamvision", "publish_runtime", "docker", "complete"]) {
    const job = jobs[name];
    for (const failed of [].concat(job.needs)) for (const result of ["failure", "skipped", "cancelled"]) {
      const ctx = context();
      for (const dep of [].concat(job.needs)) ctx.needs[dep] = { result: "success", outputs: { channel: "rc" } };
      ctx.needs[failed].result = result;
      equal(jobEnabled(job, ctx), false, `${name}: ${failed}/${result} blocks publication`);
      scenarios++;
    }
    if (!name.startsWith("publish_")) continue;
    equal(job.permissions, { contents: "read", "id-token": "write" }, "PyPI permission boundary");
    equal(job.uses, undefined, "PyPI upload stays in trusted root workflow");
    equal(job.environment, undefined, "preserve trusted publisher environment");
    const upload = job.steps.find(step => step.uses?.startsWith("pypa/"));
    const verify = job.steps.find(step => /(?:publish.sh verify |release.sh verify-sdk )/.test(step.run || ""));
    const prepare = job.steps.find(step => step.id === "prepare");
    assert.ok(upload && verify && prepare, "publication preparation, upload, verification all required");
    assert.ok(job.steps.indexOf(prepare) < job.steps.indexOf(upload) && job.steps.indexOf(upload) < job.steps.indexOf(verify));
    equal(upload["continue-on-error"], true, "reconcile partial/duplicate uploads");
    equal(verify["continue-on-error"] ?? false, false, "strict verification is final authority");
    equal(upload.with["skip-existing"] ?? false, false, "never blindly skip duplicates");
    for (const missing of ["true", "false", "", undefined]) for (const previous of results) for (const cancelled of [false, true]) {
      const ctx = context(); ctx.steps.prepare = { outputs: { upload: missing } };
      const ready = !["failure", "cancelled"].includes(previous) && !cancelled;
      equal(stepEnabled(upload, ctx, previous, cancelled), ready && missing === "true", "only verified missing artifacts upload");
      equal(stepEnabled(verify, ctx, previous, cancelled), ready, "verify no-op and successful uploads");
      scenarios++;
    }
    for (const uploaded of ["success", "failure"]) for (const verified of ["success", "failure"]) {
      const execution = runSteps(job, context(), new Map([[upload, uploaded], [verify, verified]]));
      equal(execution.ran.includes(verify), true, "failed upload reaches content reconciliation");
      equal(execution.result, verified, "verified complete inventory decides success");
      scenarios++;
    }
  }
  const runtime = jobs.publish_runtime.steps;
  const gate = runtime.find(step => step.run?.includes("dependencies.sh verify"));
  assert.ok(gate?.run.includes("gate-sdk"), "runtime retries reverify all dependencies");
  equal(gate["continue-on-error"] ?? false, false, "dependency mismatch blocks runtime");
  assert.ok(runtime.indexOf(gate) < runtime.findIndex(step => step.id === "prepare"));
}

function checkChildren(children) {
  for (const [name, { directory, workflow: wf }] of Object.entries(children)) {
    const job = wf.jobs.build;
    equal(job["timeout-minutes"], 50, "child includes dependency gate, content verification and build/upload margin");
    const budget = job.steps.flatMap(step => [...(step.run || "").matchAll(/--wait-seconds (\d+)/g)]).reduce((sum, match) => sum + Number(match[1]), 0);
    equal(budget, 3 * 600, "child dependency, preparation and final verification budgets");
    equal(job["timeout-minutes"] * 60 - budget, 1200, "twenty-minute child build and upload allowance");

    const upload = job.steps.find(step => step.uses?.startsWith("pypa/"));
    equal(upload.uses, "pypa/gh-action-pypi-publish@7245906fb61fa33a13554f9d5fb856fe61758086", "verified PyPA action pin");
    const verification = job.steps[job.steps.indexOf(upload) + 1];
    assert.ok(verification.run?.includes(`publish.sh verify --profile ${name} --directory ${directory} `), "verify built candidate immediately after original upload");
    assert.ok(verification.run.includes("--receipt release-evidence/publication-receipt.json") && verification.run.includes("--wait-seconds 600"), "retain content evidence within ten-minute budget");
    equal(verification["continue-on-error"] ?? false, false, "child content verification is strict");
    assert.ok(verification.if.includes("github.event.inputs.publish == 'true'"), "every publishing child verifies contents");
    assert.ok(!verification.if.includes("steps.prepare"), "no-op publication still verifies");
    const preparation = job.steps.find(step => step.id === "prepare");
    assert.ok(preparation?.run.includes(`publish.sh prepare --profile ${name} --directory ${directory} `));
    assert.ok(job.steps.indexOf(preparation) < job.steps.indexOf(upload));
    equal(upload.with, { "packages-dir": "release-upload/" }, "upload only staged distributions");
    equal(upload["continue-on-error"], true, "child upload races reach strict verification");
    for (const uploadResult of ["success", "failure"]) for (const verificationResult of ["success", "failure"]) {
      const ctx = context();
      ctx.github.event.inputs = { publish: "true" };
      const execution = runSteps(job, ctx, new Map([[upload, uploadResult], [verification, verificationResult]]));
      equal(execution.ran.includes(verification), true, "child verification runs after upload failure");
      equal(execution.result, verificationResult, "complete verified inventory controls child result");
      scenarios++;
    }
    for (const publish of ["false", "true"]) for (const missing of ["false", "true", "", undefined]) for (const previous of results) for (const cancelled of [false, true]) {
      const ctx = context();
      ctx.github.event.inputs = { publish };
      ctx.steps.prepare = { outputs: { upload: missing } };
      const ready = publish === "true" && !["failure", "cancelled"].includes(previous) && !cancelled;
      equal(stepEnabled(preparation, ctx, previous, cancelled), ready, "only publishing children prepare");
      equal(stepEnabled(upload, ctx, previous, cancelled), ready && missing === "true", "only missing verified distributions upload");
      equal(stepEnabled(verification, ctx, previous, cancelled), ready, "complete inventory is verified for uploads and no-ops");
      scenarios++;
    }

    for (const publish of ["false", "true"]) for (const previous of results) for (const cancelled of [false, true]) {
      const ctx = context();
      ctx.github.event.inputs = { publish };
      equal(stepEnabled(verification, ctx, previous, cancelled), publish === "true" && !["failure", "cancelled"].includes(previous) && !cancelled, "child verification respects publication, prior failures and cancellation");
      scenarios++;
    }
    const ctx = context();
    ctx.github.event.inputs = { publish: "true" };
    const execution = runSteps(job, ctx, new Map([[verification, "failure"]]));
    equal(execution.result, "failure", "content mismatch makes exact publisher run unsuccessful");
    assert.ok(execution.ran.includes(verification), "published content check runs");
    scenarios++;
  }
}

function checkImage(wf) {
  equal(wf["cache-mode"], "none", "image workflow cannot access shared Actions caches");
  for (const job of Object.values(wf.jobs)) equal(job["cache-mode"] ?? "none", "none", "image jobs cannot restore cache access");
  const build = wf.jobs.build;
  const step = id => build.steps.find(s => s.id === id);
  for (const bake of [false, true]) for (const reused of ["", "sha256:" + "a".repeat(64)]) {
    const ctx = context();
    ctx.inputs.image = JSON.stringify({ bake });
    ctx.steps = { plan: { outputs: { bake: JSON.stringify(bake), reuse_digest: reused } }, image: { outputs: { digest: reused ? "" : "sha256:" + "b".repeat(64) } } };
    equal(stepEnabled(step("image"), ctx), !bake && !reused, "only build missing non-bake images");
    equal(stepEnabled(step("bake"), ctx), bake && !reused, "only bake missing images");
    equal(evaluate(step("receipt").env.IMAGE_DIGEST, ctx), reused || ctx.steps.image.outputs.digest, "retain reused digest without rebuilding");
    scenarios++;
  }
  equal([].concat(wf.jobs.publish.needs), ["build", "smoke"], "promotion requires target-platform smoke");
  for (const s of build.steps) {
    if (s.uses && !/^(actions\/|\.\/)/.test(s.uses)) assert.match(s.uses, /@[0-9a-f]{40}$/, "third-party publishing actions are immutable");
  }
  for (const [input, producer] of [["runtime_artifact", "build_runtime"], ["sdk_artifact", "build_sdk"]]) {
    equal(workflow.jobs.docker.with[input], `\${{ needs.${producer}.outputs.artifact }}`, "image finalization uses retained published wheels");
    assert.ok(workflow.jobs.docker.needs.includes(producer));
    equal(build.steps.find(s => s.with?.name === `\${{ inputs.${input} }}`).with.path, "dist", "retained wheel extraction root");
  }
  for (const name of ["smoke", "publish"]) {
    const download = wf.jobs[name].steps.find(s => s.uses?.startsWith("actions/download-artifact@"));
    equal(download.with.name, "${{ needs.build.outputs.plan_artifact }}", "consumer uses retained image plan");
    equal(download.with.path, undefined, "plan and digest extract into working directory");
  }
}

const imageWorkflow = parse(fs.readFileSync("/work/.github/workflows/release.image.yml", "utf8"));
checkImage(imageWorkflow);
check(workflow);
checkChildren(upstreams);
console.log(`Workflow conditions: ${checks} checks passed across ${scenarios} scenarios`);
const mutations = [
  ["extract SDK into wrong root", wf => { wf.jobs.publish_sdk.steps.find(s => s.uses?.startsWith("actions/download-artifact@")).with.path = "dist/sdk"; }],
  ["lose artifact producer dependency", wf => { wf.jobs.build_models.steps.find(s => s.uses?.startsWith("actions/download-artifact@")).with.name = "${{ needs.build_sdk.outputs.artifact }}"; }],
  ["restore release cache writes", wf => { wf["cache-mode"] = "write"; }],
  ["override release cache restriction", wf => { wf.jobs.build_runtime["cache-mode"] = "write"; }],
  ["inherit runtime version for Models", wf => { wf.jobs.publish_models.env.EXPECTED_VERSION = "${{ inputs.expected_version }}"; }],
  ["publish on created", wf => { wf.on.release.types = ["created"]; }],
  ["skip RC validation", wf => { wf.jobs.prepare_dependencies.needs = wf.jobs.prepare_dependencies.needs.filter(name => name !== "validation"); }],
  ["run full stable tests", wf => { wf.jobs.rc_unit_tests_inference_x86.if = "${{ true }}"; }],
  ["test stale SHA", wf => { wf.jobs.rc_unit_tests_inference_x86.with.release_source = "${{ github.sha }}"; }],
  ["dispatch extra runs", wf => { wf.jobs.prepare_dependencies.steps.push({run: "bash .release/pypi/dependencies.sh coordinate"}); }],
  ["publish after failed gates", wf => { wf.jobs.publish_runtime.if = "${{ always() }}"; }],
  ["unconditional runtime upload", wf => { delete wf.jobs.publish_runtime.steps.find(step => step.uses?.startsWith("pypa/")).if; }],
  ["drop runtime verification", wf => { wf.jobs.publish_runtime.steps = wf.jobs.publish_runtime.steps.filter(step => !step.run?.includes("publish.sh verify ")); }],
  ["tolerate runtime mismatch", wf => { wf.jobs.publish_runtime.steps.find(step => step.run?.includes("publish.sh verify "))["continue-on-error"] = true; }],
  ["drop runtime dependency gate", wf => { wf.jobs.publish_runtime.steps = wf.jobs.publish_runtime.steps.filter(step => !step.run?.includes("dependencies.sh verify")); }],
  ["stop on partial SDK upload", wf => { delete wf.jobs.publish_sdk.steps.find(step => step.uses?.startsWith("pypa/"))["continue-on-error"]; }],
  ["publish previews", wf => { wf.jobs.publish_sdk.if = "${{ true }}"; }],
  ["PUBLISH hardcoded true", wf => { wf.env.PUBLISH = "${{ true }}"; }],
  ["ignore release tag", wf => { wf.env.EXPECTED_VERSION = "${{ inputs.expected_version }}"; }],
  ["publish forks", wf => { delete wf.jobs.prepare.if; }],
];
for (const [name, mutate] of mutations) {
  const faulty = structuredClone(workflow);
  mutate(faulty);
  assert.notDeepEqual(faulty, workflow, `${name}: mutation applied`);
  assert.throws(() => check(faulty), assert.AssertionError, `${name}: rejected`);
  console.log(`Mutation rejected: ${name}`);
}
console.log(`Workflow mutations: ${mutations.length}/${mutations.length} rejected`);

const childMutations = [
  ["stop on child duplicate upload", job => { delete job.steps.find(step => step.uses?.startsWith("pypa/"))["continue-on-error"]; }],
  ["unconditional child upload", job => { delete job.steps.find(step => step.uses?.startsWith("pypa/")).if; }],
  ["drop child preparation", job => { job.steps = job.steps.filter(step => step.id !== "prepare"); }],
  ["remove content verification", job => { job.steps = job.steps.filter(step => !step.run?.includes("publish.sh verify ")); }],
  ["tolerate content mismatch", job => { job.steps.find(step => step.run?.includes("publish.sh verify "))["continue-on-error"] = true; }],
  ["skip content verification", job => { job.steps.find(step => step.run?.includes("publish.sh verify ")).if = "${{ false }}"; }],
];
for (const [name, mutate] of childMutations) {
  const faulty = structuredClone(upstreams);
  mutate(faulty["inference-models"].workflow.jobs.build);
  assert.notDeepEqual(faulty, upstreams, `${name}: mutation applied`);
  assert.throws(() => checkChildren(faulty), assert.AssertionError, `${name}: rejected`);
  console.log(`Child mutation rejected: ${name}`);
}
console.log(`Child workflow mutations: ${childMutations.length}/${childMutations.length} rejected`);

const imageMutations = [
  ["allow image cache writes", wf => { wf["cache-mode"] = "write"; }],
  ["rebuild unchanged Models image", wf => { delete wf.jobs.build.steps.find(s => s.id === "image").if; }],
  ["lose reused image digest", wf => { wf.jobs.build.steps.find(s => s.id === "receipt").env.IMAGE_DIGEST = "${{ steps.image.outputs.digest }}"; }],
  ["publish images before smoke", wf => { wf.jobs.publish.needs = ["build"]; }],
];
for (const [name, mutate] of imageMutations) {
  const faulty = structuredClone(imageWorkflow);
  mutate(faulty);
  assert.notDeepEqual(faulty, imageWorkflow, `${name}: mutation applied`);
  assert.throws(() => checkImage(faulty), assert.AssertionError, `${name}: rejected`);
  console.log(`Image mutation rejected: ${name}`);
}
console.log(`Image workflow mutations: ${imageMutations.length}/${imageMutations.length} rejected`);
