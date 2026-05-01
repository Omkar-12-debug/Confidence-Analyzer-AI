use std::sync::Mutex;
use tauri::Manager;
use tauri_plugin_shell::ShellExt;
use tauri_plugin_shell::process::{CommandChild, CommandEvent};

/// Holds the backend child-process handle so it is killed when the app exits.
struct BackendProcess(Mutex<Option<CommandChild>>);

impl Drop for BackendProcess {
    fn drop(&mut self) {
        if let Ok(mut guard) = self.0.lock() {
            if let Some(child) = guard.take() {
                let _ = child.kill();
                println!("[Tauri] Backend sidecar killed on app exit");
            }
        }
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_opener::init())
        .setup(|app| {
            // ---------- Launch Python backend sidecar ----------
            let shell = app.shell();

            match shell.sidecar("run_backend") {
                Ok(sidecar_cmd) => {
                    match sidecar_cmd.spawn() {
                        Ok((mut rx, child)) => {
                            println!("[Tauri] Backend sidecar spawned successfully");

                            // Keep the child handle alive so Drop kills it on exit.
                            app.manage(BackendProcess(Mutex::new(Some(child))));

                            // Forward sidecar stdout/stderr to the host console.
                            tauri::async_runtime::spawn(async move {
                                while let Some(event) = rx.recv().await {
                                    match event {
                                        CommandEvent::Stdout(line) => {
                                            let msg = String::from_utf8_lossy(&line);
                                            print!("[Backend] {}", msg);
                                        }
                                        CommandEvent::Stderr(line) => {
                                            let msg = String::from_utf8_lossy(&line);
                                            eprint!("[Backend] {}", msg);
                                        }
                                        CommandEvent::Terminated(payload) => {
                                            println!(
                                                "[Backend] Process terminated (code={:?}, signal={:?})",
                                                payload.code, payload.signal
                                            );
                                            break;
                                        }
                                        _ => {}
                                    }
                                }
                            });
                        }
                        Err(e) => {
                            eprintln!("[Tauri] Could not spawn backend sidecar: {e}");
                            eprintln!("[Tauri] In dev mode, start the backend manually:");
                            eprintln!("[Tauri]   python run_backend.py");
                        }
                    }
                }
                Err(e) => {
                    // Expected in dev mode when the sidecar exe has not been built yet.
                    eprintln!("[Tauri] Sidecar binary not found: {e}");
                    eprintln!("[Tauri] In dev mode, start the backend manually:");
                    eprintln!("[Tauri]   python run_backend.py");
                }
            }

            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
