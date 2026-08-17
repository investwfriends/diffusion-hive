# dsh-quickpod

QuickPod GPU/CPU cloud plugin for DeepSeek Harness (DSH). It adds model-facing
tools that search, connect, deploy, and monitor QuickPod pods — including
monitoring an AI model training session.

QuickPod is a peer-to-peer GPU cloud. This plugin wraps the same public and
authenticated endpoints the official QuickPod CLI uses (see
https://docs.quickpod.io/api-and-cli/api-documentation).

## Install

Install into whichever profile you run (web, headless, tui, or a custom one):

    dsh plugin --profile web add ./dsh-quickpod

The package declares a "dsh.bundle" patch, so "dsh plugin" auto-appends it to
the profile's bundle list and it activates on next boot. Restart the profile to
load it.

## Troubleshooting

**Boot fails with `Cannot find package '@deepseek-ai/dsh-tools' imported from .../dsh-quickpod/lib/index.js`**

`dsh plugin add ./dsh-quickpod` installs the plugin as a pnpm `link:` dependency,
which does **not** install the package's dependencies at the linked location. The
plugin imports `@deepseek-ai/dsh-tools` (a peerDependency), so it must be
resolvable from the plugin's real path.

Fix — give the plugin its own `node_modules` pointing at the harness's installed
`@deepseek-ai` scope (the profile mirror, which stays in sync with the runtime
tree):

    mkdir -p node_modules
    ln -s ~/.dsh/profiles/node_modules/@deepseek-ai node_modules/@deepseek-ai

Then restart the profile. Re-run this if you move the checkout.

## Configure credentials

Either export a credential before starting dsh:

    export QUICKPOD_API_KEY=qpk_...        # secure API key (recommended)
    export QUICKPOD_TOKEN=...              # or a bearer token

or store one at runtime with the connect tool. The plugin persists the
credential to $DSH_HOME/quickpod.json (or ~/.quickpod.json when DSH_HOME is
unset), mode 0600.

Optional overrides:

    QUICKPOD_BASE_URL    default: https://api.quickpod.org

## Tools

    quickpod_search     search rentable (or occupied) GPU/CPU offers — public
    quickpod_connect    store a credential or log in, verify /update/auth/me
    quickpod_templates  list templates (public/community/my) -> template_uuid
    quickpod_deploy     create a pod (template_uuid + offer_id + disk)
    quickpod_pods       list your pods with state/status
    quickpod_status     describe one pod by uuid / id / name
    quickpod_logs       fetch a pod's logs (training output)
    quickpod_control    start / stop / restart / destroy a pod
    quickpod_wait       poll a pod until it reaches a target state

## Example: train and monitor

    quickpod_search     kind=gpu gpu_type=A100 max_hourly=2.5
    quickpod_connect    credential=qpk_...
    quickpod_templates  scope=public kind=gpu
    quickpod_deploy     kind=gpu template_uuid=<uuid> offer_id=<id> disk_size_gb=50 name=trainer
    quickpod_wait       pod=<pod_uuid> kind=gpu target_state=running
    quickpod_status     pod=<pod_uuid> kind=gpu
    quickpod_logs       pod=<pod_uuid> kind=gpu
    quickpod_control    action=destroy pod=<pod_uuid> kind=gpu

Reuse the pod_uuid returned by quickpod_deploy (or quickpod_pods) for
quickpod_status, quickpod_logs, quickpod_control, and quickpod_wait.

## API surface wrapped

Base URL: https://api.quickpod.org

Auth: credentials starting with qpk_ are sent as X-API-Key plus
Authorization: ApiKey; everything else as Authorization: Bearer.

    GET  /rentable            public GPU offers
    GET  /rentable_cpu        public CPU offers
    GET  /notrentable         occupied GPU offers
    GET  /notrentable_cpu     occupied CPU offers
    GET  /public_templates    public GPU templates
    GET  /community_templates community GPU templates
    GET  /templates           your GPU templates (auth)
    GET  /templates_cpu*      CPU template variants
    GET  /mypods              your GPU pods (auth)
    GET  /mypods_cpu          your CPU pods (auth)
    POST /update/createpod    create GPU pod (auth)
    POST /update/createpod_cpu create CPU pod (auth)
    GET  /update/startpod|stoppod|restartpod|destroypod|podlogs (auth, ?pod_uuid=)
    POST /update/auth/login   email/password login
    GET  /update/auth/me      authenticated profile

## Layout

    lib/client.js      QuickPodClient + config persistence
    lib/index.js       Cordis plugin registering the tools
    cordis.patch.yml   bundle patch inserting the plugin row
    package.json       dsh.bundle declaration

## Notes

- Deploy requires both a template_uuid (quickpod_templates) and an offer_id
  (quickpod_search). The disk size and offer determine billing.
- quickpod_wait polls /mypods and therefore needs authentication.
- All QuickPod errors are surfaced as tool errors with the API's error /
  details / message field when present.
