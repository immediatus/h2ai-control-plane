use futures::future::join_all;
use h2ai_types::adapter::{AdapterError, ComputeRequest, IComputeAdapter};
use h2ai_types::chain::{ChainedRequest, ChainStep};
use h2ai_types::sizing::TauValue;

pub async fn execute_chain(
    adapter: &dyn IComputeAdapter,
    req: ChainedRequest,
) -> Result<String, AdapterError> {
    let mut ctx = req.initial_system_context;
    let mut output = String::new();

    for step in req.steps {
        let resp = adapter
            .execute(ComputeRequest {
                system_context: ctx,
                task: step.template,
                tau: step.tau,
                max_tokens: step.max_tokens,
            })
            .await?;
        output = resp.output;
        ctx = output.clone();
    }

    Ok(output)
}

pub async fn tournament_merge(
    adapter: &dyn IComputeAdapter,
    system_context: &str,
    proposals: Vec<String>,
    merge_template: &str,
    tau: TauValue,
    max_tokens: u64,
) -> Result<String, AdapterError> {
    if proposals.is_empty() {
        return Ok(String::new());
    }

    let mut current = proposals;

    while current.len() > 1 {
        let (pairs, bye) = split_pairs(current);

        let merge_futs: Vec<_> = pairs.into_iter().map(|(a, b)| {
            let sys = format!("{system_context}\n\n## Current Best:\n{a}");
            let task = merge_template.replace("{proposal_b}", &b);
            execute_chain(
                adapter,
                ChainedRequest {
                    initial_system_context: sys,
                    steps: vec![ChainStep {
                        template: task,
                        tau,
                        max_tokens,
                    }],
                },
            )
        }).collect();

        let mut next: Vec<String> = join_all(merge_futs)
            .await
            .into_iter()
            .collect::<Result<_, _>>()?;

        if let Some(b) = bye {
            next.push(b);
        }

        current = next;
    }

    Ok(current.into_iter().next().unwrap_or_default())
}

fn split_pairs(items: Vec<String>) -> (Vec<(String, String)>, Option<String>) {
    let mut pairs = Vec::new();
    let mut bye: Option<String> = None;
    let mut iter = items.into_iter();
    loop {
        match (iter.next(), iter.next()) {
            (Some(a), Some(b)) => pairs.push((a, b)),
            (Some(a), None) => {
                bye = Some(a);
                break;
            }
            _ => break,
        }
    }
    (pairs, bye)
}
