"""Generate professional PNG diagrams for AutoPredict documentation."""
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

OUT = Path(r"C:\Users\Developer\Projects\AutoPredict\docs\diagrams")
OUT.mkdir(parents=True, exist_ok=True)

# ── Palette ──────────────────────────────────────────────────────────────────
BG    = '#F8FAFC'
NAVY  = '#0F2044'
BLUE  = '#1E5799'
LBLUE = '#2E86AB'
CYAN  = '#00B4D8'
TEAL  = '#0D7377'
GREEN = '#14A44D'
ORG   = '#E76F51'
RED   = '#E63946'
PUR   = '#7B2D8B'
GRAY  = '#64748B'
LGRAY = '#E2E8F0'
DARK  = '#1A202C'
WHITE = '#FFFFFF'

def styled_fig(w=16, h=10):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.axis('off')
    return fig, ax

def rbox(ax, x, y, w, h, fc, ec, lw=1.5, radius=0.3, alpha=1.0):
    r = FancyBboxPatch((x, y), w, h,
                       boxstyle=f'round,pad={radius}',
                       linewidth=lw, edgecolor=ec, facecolor=fc, alpha=alpha,
                       zorder=3)
    ax.add_patch(r)
    return r

def arrow(ax, x0, y0, x1, y1, color=GRAY, lw=1.8, style='->', shrink=4):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=style, color=color,
                                lw=lw, shrinkA=shrink, shrinkB=shrink))

def title_block(ax, title, subtitle, xlim=16, ylim=10):
    ax.set_xlim(0, xlim)
    ax.set_ylim(0, ylim)
    ax.text(xlim/2, ylim - 0.4, title, ha='center', va='top',
            fontsize=18, fontweight='bold', color=NAVY)
    ax.text(xlim/2, ylim - 1.0, subtitle, ha='center', va='top',
            fontsize=11, color=GRAY)
    # Horizontal rule
    ax.plot([0.5, xlim - 0.5], [ylim - 1.3, ylim - 1.3], color=LGRAY, lw=1.5)

# ── Diagram 1 — System Architecture ─────────────────────────────────────────
def diag_system_arch():
    fig, ax = styled_fig(18, 12)
    title_block(ax, 'AutoPredict — System Architecture',
                'Five-Layer Predictive Maintenance Platform', xlim=18, ylim=12)

    layers = [
        dict(y=1.2, h=1.4, fc='#FFF0F0', ec='#E63946', tag='#E63946',
             name='LAYER 1 — Vehicle / Edge',
             items=['MG Vehicle Fleet\n(TBox ECU + OBD)', 'MQTT Telemetry\n200+ Signals / 5 Hz', 'GPS + Sensors\n(Speed, Temp, SOC)']),
        dict(y=3.2, h=1.4, fc='#F0E6FF', ec='#7B2D8B', tag='#7B2D8B',
             name='LAYER 2 — Messaging & Storage',
             items=['MQTT Broker\npaho-mqtt port 1883', 'CSV Data Store\npandas offline-first', 'Model Artifacts\n.pkl / .joblib']),
        dict(y=5.2, h=1.4, fc='#FFF7E0', ec='#E76F51', tag='#E76F51',
             name='LAYER 3 — ML & Business Logic',
             items=['8× LightGBM Models\n(brake, oil, tyre …)', 'Champion-Challenger\nPSI Drift Monitor', '4× EV Physics Engines\n(DCDC, motor, thermal)']),
        dict(y=7.2, h=1.4, fc='#E6F7EC', ec='#14A44D', tag='#14A44D',
             name='LAYER 4 — API Gateway',
             items=['FastAPI + Uvicorn\nport 8001', 'JWT HS256 Auth\n3-role RBAC', 'slowapi Rate Limiter\n+ CORS guard']),
        dict(y=9.2, h=1.4, fc='#E6F0FF', ec='#1E5799', tag='#1E5799',
             name='LAYER 5 — Presentation',
             items=['Dealer Portal\nReact 18 + TypeScript', 'OEM Analytics\nFleet / Model / EDA', 'Admin Panel\nUser management']),
    ]

    for L in layers:
        # Big layer container
        rbox(ax, 0.4, L['y'], 17.2, L['h'], L['fc'], L['ec'], lw=2, radius=0.15)
        ax.text(0.9, L['y'] + L['h'] - 0.28, L['name'],
                fontsize=9, fontweight='bold', color=L['tag'], va='top')
        # Three component cards
        for i, item in enumerate(L['items']):
            cx = 3.5 + i * 5.0
            rbox(ax, cx - 1.8, L['y'] + 0.12, 3.6, 0.9, WHITE, L['ec'], lw=1.2, radius=0.12)
            ax.text(cx, L['y'] + 0.57, item, ha='center', va='center',
                    fontsize=8.5, color=DARK, multialignment='center')

    # Vertical arrows between layers
    for ya in [2.7, 4.7, 6.7, 8.7]:
        arrow(ax, 9, ya, 9, ya + 0.45, color=GRAY, lw=2, style='<->', shrink=2)

    # Legend
    legend_items = [
        mpatches.Patch(facecolor='#FFF0F0', edgecolor='#E63946', label='Vehicle / Edge'),
        mpatches.Patch(facecolor='#F0E6FF', edgecolor='#7B2D8B', label='Messaging & Storage'),
        mpatches.Patch(facecolor='#FFF7E0', edgecolor='#E76F51', label='ML & Business Logic'),
        mpatches.Patch(facecolor='#E6F7EC', edgecolor='#14A44D', label='API Gateway'),
        mpatches.Patch(facecolor='#E6F0FF', edgecolor='#1E5799', label='Presentation'),
    ]
    ax.legend(handles=legend_items, loc='lower right', fontsize=8,
              framealpha=0.9, edgecolor=LGRAY, ncol=5,
              bbox_to_anchor=(0.99, 0.01))

    plt.tight_layout(pad=0.3)
    plt.savefig(OUT / 'diag_01_system_arch.png', dpi=200, bbox_inches='tight', facecolor=BG)
    plt.close()
    print('  ✓ diag_01_system_arch.png')

# ── Diagram 2 — Data Flow Pipeline ───────────────────────────────────────────
def diag_data_flow():
    fig, ax = styled_fig(20, 8)
    title_block(ax, 'AutoPredict — Data Pipeline',
                'From raw TBox telemetry to dealer alert', xlim=20, ylim=8)

    steps = [
        ('TBox\nSignal', CYAN, 1.5, 4.5),
        ('MQTT\nBroker', PUR, 4.0, 4.5),
        ('CSV\nIngestion', BLUE, 6.5, 4.5),
        ('Feature\nEngineering', TEAL, 9.0, 4.5),
        ('LightGBM\nInference', ORG, 11.5, 4.5),
        ('Risk\nScoring', RED, 14.0, 4.5),
        ('Alert +\nUI Display', GREEN, 16.5, 4.5),
    ]

    for label, color, x, y in steps:
        # Shadow
        rbox(ax, x - 1.05, y - 0.82, 2.1, 1.44, '#DDDDDD', '#BBBBBB', lw=0, radius=0.25, alpha=0.5)
        # Card
        rbox(ax, x - 1.0, y - 0.78, 2.0, 1.38, color, color, lw=0, radius=0.22)
        ax.text(x, y - 0.09, label, ha='center', va='center',
                fontsize=9, fontweight='bold', color=WHITE, multialignment='center')

    # Arrows
    for i in range(len(steps) - 1):
        x0 = steps[i][2] + 1.05
        x1 = steps[i+1][2] - 1.05
        y_ = steps[i][3] - 0.09
        arrow(ax, x0, y_, x1, y_, color=GRAY, lw=2, shrink=2)

    # Sub-labels below
    sub = ['EV/PHEV/ICE schema\n200+ signals at 5 Hz',
           'Port 1883\npaho-mqtt',
           'pandas\noffline-first',
           'Rolling windows\nlag features',
           '8 models\n4 EV engines',
           'PSI drift\nconfidence %',
           'React UI\nDealer action']
    for (_, _, x, y), s in zip(steps, sub):
        ax.text(x, y - 1.1, s, ha='center', va='top',
                fontsize=7.5, color=GRAY, multialignment='center')

    # Cold-start annotation
    ax.annotate('Cold read: 55–65 s\n(first CSV load)', xy=(6.5, 3.35), xytext=(6.5, 2.2),
                ha='center', fontsize=8, color=ORG,
                arrowprops=dict(arrowstyle='->', color=ORG, lw=1.2))

    plt.tight_layout(pad=0.3)
    plt.savefig(OUT / 'diag_02_data_flow.png', dpi=200, bbox_inches='tight', facecolor=BG)
    plt.close()
    print('  ✓ diag_02_data_flow.png')

# ── Diagram 3 — Auth & RBAC ──────────────────────────────────────────────────
def diag_auth_rbac():
    fig, ax = styled_fig(18, 11)
    title_block(ax, 'AutoPredict — Authentication & RBAC',
                'JWT HS256 · Three-tier roles · Horizontal tenant isolation', xlim=18, ylim=11)

    # ── JWT Flow (left side) ─────────────────────────────
    ax.text(4.5, 9.5, 'Authentication Flow', fontsize=12, fontweight='bold',
            ha='center', color=NAVY)

    flow = [
        (4.5, 8.6, 'POST /api/auth/token\nusername + password', BLUE, WHITE),
        (4.5, 7.2, 'Verify credentials\n(user store lookup)', TEAL, WHITE),
        (4.5, 5.8, 'Issue JWT (HS256)\nsub / role / dealer_code / exp', GREEN, WHITE),
        (4.5, 4.4, 'Client stores in\nlocalStorage (ap_token)', LBLUE, WHITE),
        (4.5, 3.0, 'Authorization: Bearer <token>\non every API request', PUR, WHITE),
        (4.5, 1.6, 'get_current_user()\ndecodes + validates JWT', ORG, WHITE),
    ]
    for x, y, label, fc, tc in flow:
        rbox(ax, x - 2.5, y - 0.48, 5.0, 0.92, fc, fc, lw=0, radius=0.18)
        ax.text(x, y, label, ha='center', va='center',
                fontsize=8.5, color=tc, multialignment='center')
    for i in range(len(flow) - 1):
        arrow(ax, flow[i][0], flow[i][1] - 0.5, flow[i+1][0], flow[i+1][1] + 0.5,
              color=GRAY, lw=1.8, shrink=3)

    # Divider
    ax.plot([9, 9], [1.0, 10.5], color=LGRAY, lw=2, linestyle='--')

    # ── RBAC Pyramid (right side) ────────────────────────
    ax.text(13.5, 9.5, 'Role Hierarchy', fontsize=12, fontweight='bold',
            ha='center', color=NAVY)

    roles = [
        (13.5, 8.2, 5.0, 'ADMIN', RED,   'Full system access\nUser management · All data'),
        (13.5, 6.6, 6.5, 'OEM',   ORG,   'Fleet-wide view · Model retraining\nEDA · What-If · All dealer data'),
        (13.5, 5.0, 8.0, 'DEALER', BLUE, 'Own fleet only (dealer_code scoped)\nDashboard · Alerts · Service Bay'),
    ]
    for x, y, w, name, fc, desc in roles:
        rbox(ax, x - w/2, y - 0.55, w, 1.0, fc, fc, lw=0, radius=0.2)
        ax.text(x, y + 0.1, name, ha='center', va='center',
                fontsize=11, fontweight='bold', color=WHITE)
        ax.text(x, y - 0.3, desc, ha='center', va='center',
                fontsize=7.5, color=WHITE, multialignment='center')

    # Hierarchy arrows
    arrow(ax, 13.5, 5.55, 13.5, 6.0, color=LGRAY, lw=1.5, style='->', shrink=2)
    arrow(ax, 13.5, 7.15, 13.5, 7.6, color=LGRAY, lw=1.5, style='->', shrink=2)

    # Tenant isolation note
    rbox(ax, 9.5, 2.5, 8.0, 1.8, '#FFF9E6', '#E76F51', lw=1.5, radius=0.2)
    ax.text(13.5, 3.7, '🔒  Horizontal Tenant Isolation', ha='center',
            fontsize=10, fontweight='bold', color=ORG)
    ax.text(13.5, 3.2,
            'dealer_code claim in JWT binds every query.\n'
            'DL001 cannot see DL002 data even with valid token.\n'
            'Enforced in every ORM/CSV filter — not just UI guards.',
            ha='center', va='center', fontsize=8, color=DARK, multialignment='center')

    # JWT structure box
    rbox(ax, 0.3, 0.3, 8.2, 1.0, '#F0F4FF', BLUE, lw=1.5, radius=0.15)
    ax.text(4.4, 0.8, 'JWT Payload: { sub, role, dealer_code, iat, exp }',
            ha='center', va='center', fontsize=8.5, color=BLUE,
            fontfamily='monospace')

    plt.tight_layout(pad=0.3)
    plt.savefig(OUT / 'diag_03_auth_rbac.png', dpi=200, bbox_inches='tight', facecolor=BG)
    plt.close()
    print('  ✓ diag_03_auth_rbac.png')

# ── Diagram 4 — Deployment Architecture ──────────────────────────────────────
def diag_deployment():
    fig, ax = styled_fig(18, 10)
    title_block(ax, 'AutoPredict — Deployment Architecture',
                'Docker Compose · Three-container stack · Development & Production', xlim=18, ylim=10)

    # Docker Compose border
    rbox(ax, 0.4, 0.4, 17.2, 7.5, '#F0F4FF', BLUE, lw=2.5, radius=0.3)
    ax.text(1.1, 7.65, 'docker-compose.yml', fontsize=9, fontweight='bold', color=BLUE)

    containers = [
        dict(x=1.0, y=1.0, w=4.5, h=5.8, fc='#E6F0FF', ec=BLUE,
             title='frontend', subtitle='React + Vite',
             port='3000', items=[
                 'React 18 + TypeScript',
                 'Vite 5 build tooling',
                 'TanStack Query v5',
                 'Recharts 2 + Tailwind',
                 'Playwright E2E suite',
             ]),
        dict(x=6.75, y=1.0, w=4.5, h=5.8, fc='#E6F7EC', ec=GREEN,
             title='backend', subtitle='FastAPI + Uvicorn',
             port='8001', items=[
                 'FastAPI + Uvicorn',
                 '6 API routers (36 routes)',
                 'LightGBM ML models',
                 'pandas CSV engine',
                 'python-jose JWT',
             ]),
        dict(x=12.5, y=1.0, w=4.5, h=5.8, fc='#FFF7E0', ec=ORG,
             title='mqtt-broker', subtitle='Eclipse Mosquitto',
             port='1883', items=[
                 'Eclipse Mosquitto',
                 'Vehicle telemetry ingest',
                 'paho-mqtt client',
                 'TBox signal decoder',
                 'CSV writer sink',
             ]),
    ]
    for c in containers:
        rbox(ax, c['x'], c['y'], c['w'], c['h'], c['fc'], c['ec'], lw=2, radius=0.2)
        # Container header
        rbox(ax, c['x'], c['y'] + c['h'] - 1.1, c['w'], 1.0, c['ec'], c['ec'], lw=0, radius=0.18)
        ax.text(c['x'] + c['w']/2, c['y'] + c['h'] - 0.52,
                c['title'], ha='center', va='center',
                fontsize=11, fontweight='bold', color=WHITE)
        ax.text(c['x'] + c['w']/2, c['y'] + c['h'] - 0.88,
                c['subtitle'], ha='center', va='center', fontsize=8, color=WHITE)
        # Port badge
        rbox(ax, c['x'] + c['w']/2 - 0.6, c['y'] + c['h'] - 1.55, 1.2, 0.38,
             WHITE, c['ec'], lw=1.2, radius=0.1)
        ax.text(c['x'] + c['w']/2, c['y'] + c['h'] - 1.36,
                f':{c["port"]}', ha='center', va='center',
                fontsize=8.5, fontweight='bold', color=c['ec'])
        # Items
        for i, item in enumerate(c['items']):
            ax.text(c['x'] + 0.35, c['y'] + 3.6 - i * 0.62, f'▸  {item}',
                    fontsize=8, color=DARK, va='center')

    # Arrows between containers
    arrow(ax, 5.55, 3.9, 6.7, 3.9, color=GRAY, lw=2, style='<->', shrink=3)
    arrow(ax, 11.3, 3.9, 12.45, 3.9, color=GRAY, lw=2, style='<->', shrink=3)

    # External: vehicle
    rbox(ax, 0.5, 8.3, 4.5, 0.9, '#FFE4E6', RED, lw=1.5, radius=0.18)
    ax.text(2.75, 8.75, 'MG Vehicle Fleet  (TBox ECU)', ha='center', va='center',
            fontsize=9, fontweight='bold', color=RED)
    arrow(ax, 14.75, 8.3, 14.75, 6.8, color=RED, lw=1.8, shrink=3)
    ax.text(15.6, 7.55, 'MQTT\ntelemetry', fontsize=7.5, color=RED, ha='center')

    # External: browser
    rbox(ax, 0.5, 8.3, 4.5, 0.9, '#FFE4E6', RED, lw=1.5, radius=0.18)
    rbox(ax, 13.0, 8.3, 4.5, 0.9, '#E6F0FF', BLUE, lw=1.5, radius=0.18)
    ax.text(15.25, 8.75, 'Browser  (Dealer / OEM / Admin)', ha='center',
            va='center', fontsize=9, fontweight='bold', color=BLUE)
    arrow(ax, 3.25, 6.85, 3.25, 8.25, color=BLUE, lw=1.8, shrink=3)
    ax.text(4.3, 7.55, 'HTTP\nport 3000', fontsize=7.5, color=BLUE, ha='center')

    plt.tight_layout(pad=0.3)
    plt.savefig(OUT / 'diag_04_deployment.png', dpi=200, bbox_inches='tight', facecolor=BG)
    plt.close()
    print('  ✓ diag_04_deployment.png')

# ── Diagram 5 — ML Pipeline ───────────────────────────────────────────────────
def diag_ml_pipeline():
    fig, ax = styled_fig(20, 10)
    title_block(ax, 'AutoPredict — ML Pipeline',
                'Champion-Challenger model serving with PSI drift monitoring', xlim=20, ylim=10)

    # Training phase
    rbox(ax, 0.3, 0.5, 7.5, 7.5, '#F0F8FF', BLUE, lw=2, radius=0.25)
    ax.text(4.05, 7.8, 'TRAINING PHASE', ha='center', fontsize=11,
            fontweight='bold', color=BLUE)

    train_steps = [
        (4.05, 6.9, 'Raw CSV Data\n(vehicle telemetry)', LBLUE),
        (4.05, 5.5, 'Feature Engineering\n(rolling windows, lag, ratios)', TEAL),
        (4.05, 4.1, 'LightGBM Training\n(8 maintenance models)', GREEN),
        (4.05, 2.7, 'Evaluation\n(AUROC, F1, Precision)', ORG),
        (4.05, 1.3, 'Model Artifact\n(.pkl + metadata)', PUR),
    ]
    for x, y, label, fc in train_steps:
        rbox(ax, x - 2.2, y - 0.45, 4.4, 0.86, fc, fc, lw=0, radius=0.18)
        ax.text(x, y, label, ha='center', va='center', fontsize=8.5,
                color=WHITE, multialignment='center')

    for i in range(len(train_steps) - 1):
        arrow(ax, train_steps[i][0], train_steps[i][1] - 0.5,
              train_steps[i+1][0], train_steps[i+1][1] + 0.5, color=GRAY, lw=1.8, shrink=3)

    # Registry (center)
    rbox(ax, 8.5, 2.5, 3.0, 5.5, '#FFF7E0', ORG, lw=2.5, radius=0.25)
    ax.text(10.0, 7.8, 'MODEL REGISTRY', ha='center', fontsize=11,
            fontweight='bold', color=ORG)
    ax.text(10.0, 7.5, '8 LightGBM + 4 EV Engines', ha='center',
            fontsize=8, color=GRAY)

    registry_items = ['brake_pad', 'engine_oil', 'tyre_wear', 'battery_12v',
                      'hv_battery_soh', 'transmission', 'cooling_system', 'driver_score']
    for i, m in enumerate(registry_items):
        y_ = 7.0 - i * 0.55
        rbox(ax, 8.7, y_ - 0.2, 2.6, 0.38, WHITE, ORG, lw=1, radius=0.1)
        ax.text(10.0, y_, m, ha='center', va='center', fontsize=7.5, color=DARK)

    # Serving phase
    rbox(ax, 12.3, 0.5, 7.3, 7.5, '#F0FFF4', GREEN, lw=2, radius=0.25)
    ax.text(15.95, 7.8, 'SERVING PHASE', ha='center', fontsize=11,
            fontweight='bold', color=GREEN)

    serve_steps = [
        (15.95, 6.9, 'Live TBox Signal\n(vehicle telemetry)', CYAN),
        (15.95, 5.5, 'Feature Vector\n(real-time engineering)', TEAL),
        (15.95, 4.1, 'Model Inference\n(champion model)', GREEN),
        (15.95, 2.7, 'PSI Drift Monitor\n(challenger comparison)', ORG),
        (15.95, 1.3, 'Alert / API Response\n(JSON to dealer UI)', BLUE),
    ]
    for x, y, label, fc in serve_steps:
        rbox(ax, x - 2.2, y - 0.45, 4.4, 0.86, fc, fc, lw=0, radius=0.18)
        ax.text(x, y, label, ha='center', va='center', fontsize=8.5,
                color=WHITE, multialignment='center')

    for i in range(len(serve_steps) - 1):
        arrow(ax, serve_steps[i][0], serve_steps[i][1] - 0.5,
              serve_steps[i+1][0], serve_steps[i+1][1] + 0.5, color=GRAY, lw=1.8, shrink=3)

    # Arrows: training → registry → serving
    arrow(ax, 6.5, 1.3, 8.45, 1.3, color=PUR, lw=2.2, shrink=3)
    arrow(ax, 11.55, 4.1, 12.25, 4.1, color=GREEN, lw=2.2, shrink=3)

    # PSI retrain loop
    ax.annotate('', xy=(10.0, 2.5), xytext=(15.95, 2.25),
                arrowprops=dict(arrowstyle='->', color=RED, lw=2,
                                connectionstyle='arc3,rad=0.4'))
    ax.text(13.2, 1.2, 'PSI > threshold\n→ trigger retrain', ha='center',
            fontsize=8, color=RED, style='italic')

    plt.tight_layout(pad=0.3)
    plt.savefig(OUT / 'diag_05_ml_pipeline.png', dpi=200, bbox_inches='tight', facecolor=BG)
    plt.close()
    print('  ✓ diag_05_ml_pipeline.png')

# ── Diagram 6 — API Router Map ────────────────────────────────────────────────
def diag_api_routes():
    fig, ax = styled_fig(18, 10)
    title_block(ax, 'AutoPredict — API Architecture',
                'FastAPI routers · 36 endpoints · Role-gated access', xlim=18, ylim=10)

    # Central FastAPI hub
    rbox(ax, 7.5, 4.0, 3.0, 1.6, NAVY, NAVY, lw=0, radius=0.25)
    ax.text(9.0, 4.8, 'FastAPI', ha='center', fontsize=12, fontweight='bold', color=WHITE)
    ax.text(9.0, 4.3, 'port 8001', ha='center', fontsize=9, color=CYAN)

    routers = [
        dict(x=1.8,  y=8.2, label='/api/auth', sub='2 endpoints\nPOST token · GET me', fc='#E63946', endpoints=2, role='Public'),
        dict(x=5.2,  y=8.2, label='/api/fleet', sub='6 endpoints\nDashboard · KPIs · Vehicles', fc='#1E5799', endpoints=6, role='DEALER'),
        dict(x=9.0,  y=8.2, label='/api/dealer', sub='4 endpoints\nAlerts · Service Bay', fc='#2E86AB', endpoints=4, role='DEALER'),
        dict(x=12.8, y=8.2, label='/api/inventory', sub='3 endpoints\nParts · Demand · Reorder', fc='#0D7377', endpoints=3, role='DEALER'),
        dict(x=3.5,  y=1.5, label='/api/oem', sub='6 endpoints\nFleet · Models · EDA · Retrain', fc='#E76F51', endpoints=6, role='OEM'),
        dict(x=9.0,  y=1.5, label='/api/admin', sub='5 endpoints\nUsers · Create · Delete', fc='#7B2D8B', endpoints=5, role='ADMIN'),
        dict(x=14.5, y=1.5, label='/api/upload', sub='1 endpoint\nCSV batch upload', fc='#14A44D', endpoints=1, role='OEM'),
    ]

    for r in routers:
        rbox(ax, r['x'] - 1.7, r['y'] - 0.7, 3.4, 1.6, r['fc'], r['fc'], lw=0, radius=0.22)
        ax.text(r['x'], r['y'] + 0.52, r['label'], ha='center', va='center',
                fontsize=9, fontweight='bold', color=WHITE)
        ax.text(r['x'], r['y'] + 0.02, r['sub'], ha='center', va='center',
                fontsize=7.5, color=WHITE, multialignment='center', alpha=0.9)
        # Role badge
        rbox(ax, r['x'] - 0.55, r['y'] - 0.6, 1.1, 0.3, WHITE, r['fc'], lw=1, radius=0.08)
        ax.text(r['x'], r['y'] - 0.44, r['role'], ha='center', va='center',
                fontsize=7, fontweight='bold', color=r['fc'])
        # Arrow to/from center
        cx, cy = 9.0, 4.8
        dx = cx - r['x']; dy = cy - r['y']
        dist = (dx**2 + dy**2) ** 0.5
        ux, uy = dx/dist, dy/dist
        arrow(ax, r['x'] + ux*1.8, r['y'] + uy*0.85,
              cx - ux*1.6, cy - uy*0.9,
              color='#94A3B8', lw=1.5, style='<->', shrink=2)

    plt.tight_layout(pad=0.3)
    plt.savefig(OUT / 'diag_06_api_routes.png', dpi=200, bbox_inches='tight', facecolor=BG)
    plt.close()
    print('  ✓ diag_06_api_routes.png')

# ── Diagram 7 — User Journey Swimlane ────────────────────────────────────────
def diag_user_journey():
    fig, ax = styled_fig(22, 10)
    title_block(ax, 'AutoPredict — Dealer User Journey',
                'Rajesh Kumar · Service Manager · Predictive maintenance alert to bay booking', xlim=22, ylim=10)

    lanes = [
        ('#E6F0FF', BLUE,  'Rajesh\n(Dealer)'),
        ('#E6F7EC', GREEN, 'React\nFrontend'),
        ('#FFF7E0', ORG,   'FastAPI\nBackend'),
        ('#F0E6FF', PUR,   'ML Engine\n& CSV'),
    ]
    lane_h = 1.6
    lane_y_starts = [7.8, 6.1, 4.4, 2.7]

    for (fc, ec, label), y in zip(lanes, lane_y_starts):
        rbox(ax, 0.2, y - 0.05, 21.6, lane_h, fc, ec, lw=1.5, radius=0.1, alpha=0.7)
        ax.text(0.8, y + lane_h/2 - 0.05, label, ha='center', va='center',
                fontsize=8.5, fontweight='bold', color=ec, multialignment='center')

    steps = [
        (2.5,  'Login\nportal',       [(0, 7.8+0.7), (1, 6.1+0.7)],             BLUE),
        (5.0,  'View\nDashboard',     [(1, 6.1+0.7), (2, 4.4+0.7)],             GREEN),
        (7.5,  'ML scores\ncomputed', [(2, 4.4+0.7), (3, 2.7+0.7), (1, 6.1+0.7)], ORG),
        (10.0, 'Alert fires\n(brake pad)',  [(1, 6.1+0.7)],                      RED),
        (12.5, 'Open Alerts\npage',   [(0, 7.8+0.7), (1, 6.1+0.7), (2, 4.4+0.7)], BLUE),
        (15.0, 'Filter critical\ncheck VIN', [(0, 7.8+0.7), (1, 6.1+0.7)],      PUR),
        (17.5, 'Open Service\nBay',   [(0, 7.8+0.7), (1, 6.1+0.7), (2, 4.4+0.7)], TEAL),
        (20.0, 'Book bay\nappointment', [(0, 7.8+0.7)],                           GREEN),
    ]

    cols = [BLUE, GREEN, ORG, RED, BLUE, PUR, TEAL, GREEN]
    for (sx, label, conn_list, _), col in zip(steps, cols):
        rbox(ax, sx - 0.9, 8.3, 1.8, 0.85, col, col, lw=0, radius=0.15)
        ax.text(sx, 8.73, label, ha='center', va='center',
                fontsize=7.5, color=WHITE, multialignment='center', fontweight='bold')
        # Number
        ax.text(sx, 9.35, str(steps.index((sx, label, conn_list, _)) + 1),
                ha='center', va='center', fontsize=8, fontweight='bold', color=col)

    # Vertical connection lines in swimlanes
    lane_cx = {'dealer': 0.7, 'frontend': 6.9, 'backend': 5.2, 'ml': 3.5}
    step_xs = [s[0] for s in steps]
    for i, sx in enumerate(step_xs):
        dot_color = cols[i]
        for lane_y in [8.55, 6.95, 5.2, 3.5]:
            if i in [1, 2, 4, 6]:  # steps that go through backend
                ax.plot(sx, lane_y, 'o', color=dot_color, markersize=7, zorder=5)

    # Timeline bar
    ax.plot([1.5, 21.5], [2.25, 2.25], color=LGRAY, lw=2)
    for i, sx in enumerate(step_xs):
        ax.plot(sx, 2.25, 'D', color=cols[i], markersize=8, zorder=5)
    ax.text(11.5, 1.85, 'Time →', ha='center', fontsize=9, color=GRAY, style='italic')

    plt.tight_layout(pad=0.3)
    plt.savefig(OUT / 'diag_07_user_journey.png', dpi=200, bbox_inches='tight', facecolor=BG)
    plt.close()
    print('  ✓ diag_07_user_journey.png')

# ── Diagram 8 — Product Roadmap ───────────────────────────────────────────────
def diag_roadmap():
    fig, ax = styled_fig(22, 10)
    title_block(ax, 'AutoPredict — Product Roadmap',
                'Five-phase delivery plan · MVP → Enterprise scale', xlim=22, ylim=10)

    phases = [
        ('Phase 1\nFoundation', 'Q1 2024', 1.5, 3.5, '#14A44D', 'DONE',
         ['FastAPI backend', 'CSV offline engine', 'JWT auth + RBAC', 'Basic dealer UI']),
        ('Phase 2\nML Core', 'Q2 2024', 5.0, 3.5, '#1E5799', 'DONE',
         ['8 LightGBM models', 'Feature engineering', 'Model registry', 'Alert engine']),
        ('Phase 3\nOEM Portal', 'Q3 2024', 8.5, 3.5, '#E76F51', 'IN PROGRESS',
         ['Fleet analytics', 'EDA + What-If', 'Retrain control', 'EV health suite']),
        ('Phase 4\nEnterprise', 'Q4 2024', 12.0, 3.5, '#7B2D8B', 'PLANNED',
         ['PostgreSQL migration', 'Real-time WebSocket', 'Mobile PWA', 'Multi-OEM support']),
        ('Phase 5\nScale', 'Q1 2025', 15.5, 3.5, '#0D7377', 'V2',
         ['Federated ML', 'Edge inference', 'Marketplace APIs', 'ISO 21434 cert']),
    ]

    status_colors = {'DONE': '#14A44D', 'IN PROGRESS': '#E76F51', 'PLANNED': '#7B2D8B', 'V2': '#0D7377'}

    for name, qtr, x, y, fc, status, items in phases:
        # Phase card
        rbox(ax, x - 1.7, y - 2.8, 3.4, 6.2, fc + '18', fc, lw=2, radius=0.25)
        # Header
        rbox(ax, x - 1.7, y + 3.1, 3.4, 0.95, fc, fc, lw=0, radius=0.2)
        ax.text(x, y + 3.62, name, ha='center', va='center',
                fontsize=10, fontweight='bold', color=WHITE, multialignment='center')
        # Quarter
        ax.text(x, y + 2.65, qtr, ha='center', va='center', fontsize=9, color=fc, fontweight='bold')
        # Status badge
        sc = status_colors[status]
        rbox(ax, x - 0.85, y + 2.1, 1.7, 0.38, sc + '22', sc, lw=1.2, radius=0.1)
        ax.text(x, y + 2.3, status, ha='center', va='center',
                fontsize=7.5, fontweight='bold', color=sc)
        # Items
        for i, item in enumerate(items):
            iy = y + 1.4 - i * 0.88
            ax.plot(x - 1.3, iy, 's', color=fc, markersize=6)
            ax.text(x - 1.05, iy, item, va='center', fontsize=8, color=DARK)

        # Progress indicator at bottom
        progress = {'DONE': 1.0, 'IN PROGRESS': 0.6, 'PLANNED': 0.0, 'V2': 0.0}[status]
        bar_w = 3.0
        rbox(ax, x - 1.5, y - 2.6, bar_w, 0.28, LGRAY, LGRAY, lw=0, radius=0.1)
        if progress > 0:
            rbox(ax, x - 1.5, y - 2.6, bar_w * progress, 0.28, fc, fc, lw=0, radius=0.1)
        ax.text(x, y - 2.9, f'{int(progress*100)}% complete', ha='center',
                fontsize=7.5, color=GRAY)

    # Timeline connector
    timeline_y = 0.6
    ax.plot([0.3, 19.5], [timeline_y, timeline_y], color=LGRAY, lw=3)
    for _, _, x, *_ in phases:
        ax.plot(x, timeline_y, 'o', color=status_colors.get(phases[0][5], GRAY),
                markersize=10, zorder=5)
    for (_, _, x, _, fc, status, _) in phases:
        ax.plot(x, timeline_y, 'o', color=fc, markersize=10, zorder=5)
        ax.plot([x, x], [timeline_y, 0.95], '--', color=fc, lw=1, alpha=0.5)

    plt.tight_layout(pad=0.3)
    plt.savefig(OUT / 'diag_08_roadmap.png', dpi=200, bbox_inches='tight', facecolor=BG)
    plt.close()
    print('  ✓ diag_08_roadmap.png')

# ── Diagram 9 — Security Coverage ────────────────────────────────────────────
def diag_security():
    fig, ax = styled_fig(18, 10)
    title_block(ax, 'AutoPredict — Security Test Coverage',
                '59 tests across 8 domains · 5 confirmed vulnerabilities', xlim=18, ylim=10)

    # Bar chart of tests per domain
    domains = ['A. Auth\nBypass', 'B. JWT\nForgery', 'C. RBAC\nVertical',
               'D. Horizontal\nIsolation', 'E. Input\nValidation', 'F. Info\nDisclosure',
               'G. Transport\nSecurity', 'H. Auth\nHardening']
    counts = [9, 4, 10, 6, 8, 7, 3, 7]
    vulns  = [1, 1, 1, 1, 0, 1, 0, 1]  # confirmed vulnerabilities
    colors_bar = [RED if v else BLUE for v in vulns]

    bar_width = 1.5
    bar_x = [1.2 + i * 2.0 for i in range(8)]

    for x, cnt, col, dom in zip(bar_x, counts, colors_bar, domains):
        bar_h = cnt * 0.5
        rbox(ax, x - 0.7, 1.5, 1.4, bar_h, col + '33', col, lw=1.5, radius=0.1)
        ax.text(x, 1.5 + bar_h + 0.2, str(cnt), ha='center', va='bottom',
                fontsize=11, fontweight='bold', color=col)
        ax.text(x, 1.1, dom, ha='center', va='top', fontsize=7.5,
                color=DARK, multialignment='center')

    # Y-axis label
    ax.text(0.3, 4.5, 'Test Count', va='center', fontsize=9, color=GRAY, rotation=90)
    ax.plot([0.5, 0.5], [1.4, 7.5], color=LGRAY, lw=1.5)
    for y in [2, 4, 6, 8, 10]:
        ax.plot([0.5, 17.5], [1.5 + y*0.5 - 0.5, 1.5 + y*0.5 - 0.5],
                color=LGRAY, lw=0.8, linestyle='--', alpha=0.6)
        ax.text(0.45, 1.5 + y*0.5 - 0.5, str(y*1), ha='right', va='center', fontsize=7.5, color=GRAY)

    # Vulnerability summary table
    rbox(ax, 0.3, 7.8, 17.4, 1.5, '#FFF0F0', RED, lw=1.5, radius=0.18)
    ax.text(9.0, 9.1, '5 Confirmed Vulnerabilities Requiring Remediation Before Production',
            ha='center', va='center', fontsize=10, fontweight='bold', color=RED)

    vulns_detail = [
        ('VLN-001', 'CRITICAL', 'Default SECRET_KEY'),
        ('VLN-002', 'CRITICAL', 'Plaintext passwords'),
        ('VLN-003', 'HIGH',     'CORS wildcard'),
        ('VLN-004', 'HIGH',     'JWT sub not validated'),
        ('VLN-005', 'HIGH',     'Tenant token forgery'),
    ]
    sev_colors = {'CRITICAL': RED, 'HIGH': ORG}
    for i, (vid, sev, desc) in enumerate(vulns_detail):
        vx = 1.5 + i * 3.3
        ax.text(vx, 8.45, vid, ha='center', fontsize=8, fontweight='bold',
                color=sev_colors[sev])
        ax.text(vx, 8.15, f'[{sev}] {desc}', ha='center', fontsize=7.5,
                color=DARK, multialignment='center')

    plt.tight_layout(pad=0.3)
    plt.savefig(OUT / 'diag_09_security.png', dpi=200, bbox_inches='tight', facecolor=BG)
    plt.close()
    print('  ✓ diag_09_security.png')

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('Generating diagrams...')
    diag_system_arch()
    diag_data_flow()
    diag_auth_rbac()
    diag_deployment()
    diag_ml_pipeline()
    diag_api_routes()
    diag_user_journey()
    diag_roadmap()
    diag_security()
    print(f'\nAll diagrams saved to: {OUT}')
