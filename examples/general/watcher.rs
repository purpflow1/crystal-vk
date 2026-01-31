use std::{
    process,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

pub fn watcher(
    stop_flag: Arc<AtomicBool>,
    heartbeat: Arc<AtomicBool>,
    hang_timeout: Duration,
) -> std::thread::JoinHandle<()> {
    thread::spawn(move || {
        let mut last_heartbeat = Instant::now();

        loop {
            thread::sleep(Duration::from_millis(1000));

            // Check if we received a heartbeat recently
            if heartbeat.swap(false, Ordering::Relaxed) {
                last_heartbeat = Instant::now();
            }

            // Check if worker has hung
            if Instant::now().duration_since(last_heartbeat) > hang_timeout {
                println!("Watcher: the main thread hung!");
                println!("Watcher: Terminating process");
                process::exit(1);
            }

            // Check if we should stop watching
            if stop_flag.load(Ordering::Relaxed) {
                break;
            }
        }
    })
}
