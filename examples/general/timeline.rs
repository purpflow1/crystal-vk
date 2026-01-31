use std::{
    f32,
    time::{Duration, SystemTime},
};

pub struct Timeline {
    pub startup_time: SystemTime,
    pub last_frame: SystemTime,

    pub delta_time: Duration,

    pub frame_time_sum: Duration,
    pub frame_num: u32,
    pub average_delta_time_last_second: f32,

    pub min_delta: f32,
    pub max_delta: f32,

    min_delta_tmp: f32,
    max_delta_tmp: f32,
}

impl Timeline {
    pub fn new() -> Self {
        Self {
            startup_time: SystemTime::now(),
            last_frame: SystemTime::UNIX_EPOCH,
            delta_time: Duration::ZERO,
            frame_time_sum: Duration::ZERO,
            frame_num: 0,
            average_delta_time_last_second: 1.,
            min_delta: 1.,
            max_delta: 1.,
            min_delta_tmp: f32::MAX,
            max_delta_tmp: 0.,
        }
    }

    pub fn frame_begin(&mut self) {
        self.delta_time = SystemTime::now().duration_since(self.last_frame).unwrap();
        self.min_delta_tmp = self.min_delta_tmp.min(self.delta_time.as_secs_f32());
        self.max_delta_tmp = self.max_delta_tmp.max(self.delta_time.as_secs_f32());
        self.last_frame = SystemTime::now();
        self.frame_time_sum += self.delta_time;
        self.frame_num += 1;
        if self.frame_time_sum >= Duration::from_secs(1) {
            self.average_delta_time_last_second = 1. / self.frame_num as f32;
            self.frame_time_sum = Duration::ZERO;
            self.frame_num = 0;
            self.min_delta = self.min_delta_tmp;
            self.max_delta = self.max_delta_tmp;
            self.min_delta_tmp = f32::MAX;
            self.max_delta_tmp = 0.
        }
    }
}
