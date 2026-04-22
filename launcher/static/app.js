const projectCards = [...document.querySelectorAll(".project-panel")];
const projectStateCache = new Map();
const projectPollers = new Map();

function getField(card, field) {
  return card.querySelector(`[data-field="${field}"]`);
}

function getLaunchButton(card) {
  return card.querySelector('[data-action="launch"]');
}

function setLaunchBusy(card, busy) {
  const button = getLaunchButton(card);
  button.disabled = busy;
  button.textContent = busy ? "Launching..." : "Launch Workspace";
}

function setMessage(card, text, isError = false) {
  const messageBox = getField(card, "message");
  if (!messageBox) {
    return;
  }

  messageBox.textContent = text;
  messageBox.classList.toggle("is-error", isError);
}

function renderProject(card, project) {
  projectStateCache.set(project.id, project);
  const hintField = getField(card, "hint");
  if (hintField) {
    hintField.textContent = project.launchHint;
  }

  if (project.error) {
    stopPolling(project.id);
    setLaunchBusy(card, false);
    setMessage(card, `${project.error}${project.installHint ? ` ${project.installHint}` : ""}`, true);
    return;
  }

  if (project.reachable) {
    stopPolling(project.id);
    setLaunchBusy(card, false);
    setMessage(card, "Ready.", false);
    return;
  }

  if (project.status === "starting") {
    startPolling(project.id);
    setLaunchBusy(card, true);
    setMessage(card, "Launching...", false);
    return;
  }

  stopPolling(project.id);
  setLaunchBusy(card, false);
  setMessage(card, "Ready.", false);
}

async function refreshProjects() {
  try {
    const response = await fetch("/api/projects", { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`Dashboard status request failed with ${response.status}`);
    }

    const data = await response.json();
    data.projects.forEach((project) => {
      const card = document.querySelector(`[data-project="${project.id}"]`);
      if (card) {
        renderProject(card, project);
      }
    });
  } catch (error) {
    projectCards.forEach((card) => {
      setMessage(card, "Status unavailable.", true);
      setLaunchBusy(card, false);
    });
  }
}

async function fetchProject(projectId) {
  const response = await fetch("/api/projects", { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Workspace status request failed with ${response.status}`);
  }

  const data = await response.json();
  return data.projects.find((project) => project.id === projectId) ?? null;
}

function stopPolling(projectId) {
  const timer = projectPollers.get(projectId);
  if (timer) {
    window.clearInterval(timer);
    projectPollers.delete(projectId);
  }
}

function startPolling(projectId) {
  if (projectPollers.has(projectId)) {
    return;
  }

  const timer = window.setInterval(async () => {
    try {
      const project = await fetchProject(projectId);
      if (!project) {
        throw new Error("Workspace status is unavailable");
      }

      const card = document.querySelector(`[data-project="${projectId}"]`);
      if (card) {
        renderProject(card, project);
      }
    } catch (error) {
      const card = document.querySelector(`[data-project="${projectId}"]`);
      if (card) {
        stopPolling(projectId);
        setLaunchBusy(card, false);
        setMessage(card, `Workspace status check failed: ${error.message}`, true);
      }
    }
  }, 1500);

  projectPollers.set(projectId, timer);
}

async function launchProject(card) {
  const projectId = card.dataset.project;
  setLaunchBusy(card, true);
  setMessage(card, "Launching...", false);

  try {
    const response = await fetch(`/api/projects/${projectId}/launch`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: "{}",
    });
    if (!response.ok && response.status !== 202) {
      throw new Error(`Launch request failed with ${response.status}`);
    }

    const data = await response.json();
    renderProject(card, data.project);
    if (data.project?.status === "starting") {
      startPolling(projectId);
    }
    return data.project;
  } catch (error) {
    setLaunchBusy(card, false);
    setMessage(card, "Launch failed.", true);
    return null;
  }
}

function waitForReachable(projectId, timeoutMs = 45000) {
  return new Promise((resolve) => {
    const startedAt = Date.now();

    const check = async () => {
      try {
        const project = await fetchProject(projectId);
        if (!project) {
          resolve(null);
          return;
        }

        const card = document.querySelector(`[data-project="${projectId}"]`);
        if (card) {
          renderProject(card, project);
        }

        if (project.reachable) {
          resolve(project);
          return;
        }

        if (project.error || Date.now() - startedAt > timeoutMs) {
          resolve(project);
          return;
        }
      } catch {
        resolve(null);
        return;
      }

      window.setTimeout(check, 1500);
    };

    check();
  });
}

function wireCard(card) {
  card.querySelector('[data-action="launch"]').addEventListener("click", () => {
    launchProject(card);
  });

  card.querySelector('[data-action="open"]').addEventListener("click", async () => {
    let project = projectStateCache.get(card.dataset.project);

    if (!project?.reachable) {
      project = await launchProject(card);
      if (!project) {
        return;
      }

      if (!project.reachable) {
        project = await waitForReachable(card.dataset.project);
      }

      if (!project?.reachable) {
        return;
      }
    }

    window.open(project.url, "_blank", "noopener");
  });
}

projectCards.forEach(wireCard);
refreshProjects();
