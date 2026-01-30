use ash::{Entry, vk};

const VALIDATION_LAYERS: &[&str] = &["VK_LAYER_KHRONOS_validation"];

pub(crate) fn get_supported_validation_layers(entry: &Entry) -> Vec<[u8; 256]> {
    let mut supported_layers = Vec::new();

    let available_layers = match unsafe { entry.enumerate_instance_layer_properties() } {
        Ok(props) => props,
        Err(e) => {
            #[allow(unused_must_use)]
            {
                dbg!(e);
            }
            return vec![];
        }
    };
    for left_layer in available_layers {
        let left = match left_layer.layer_name_as_c_str() {
            Ok(cstr) => match cstr.to_str() {
                Ok(name) => name,
                Err(_) => continue,
            },
            Err(_) => continue,
        };
        for &right in VALIDATION_LAYERS {
            if left == right {
                let mut uname = [0u8; vk::MAX_EXTENSION_NAME_SIZE];
                let vec = left_layer
                    .layer_name
                    .iter()
                    .map(|byte| *byte as u8)
                    .collect::<Vec<_>>();
                uname.clone_from_slice(&vec);
                supported_layers.push(uname)
            }
        }
    }

    supported_layers
}
