
use sea_orm_migration::prelude::*;

#[derive(DeriveMigrationName)]
pub struct Migration;

#[async_trait::async_trait]
impl MigrationTrait for Migration {
    async fn up(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        create_market_snapshot(manager).await?;
        create_decision(manager).await?;
        create_outcome(manager).await?;
        create_model_version(manager).await?;
        create_model_evaluation(manager).await?;
        create_event(manager).await?;
        create_graph_edge(manager).await?;
        create_alert(manager).await?;
        create_setting(manager).await?;
        Ok(())
    }

    async fn down(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        manager
            .drop_table(Table::drop().table(AlgoFrameSetting::Table).if_exists().to_owned())
            .await?;
        manager
            .drop_table(Table::drop().table(AlgoFrameAlert::Table).if_exists().to_owned())
            .await?;
        manager
            .drop_table(Table::drop().table(AlgoFrameGraphEdge::Table).if_exists().to_owned())
            .await?;
        manager
            .drop_table(Table::drop().table(AlgoFrameEvent::Table).if_exists().to_owned())
            .await?;
        manager
            .drop_table(
                Table::drop()
                    .table(AlgoFrameModelEvaluation::Table)
                    .if_exists()
                    .to_owned(),
            )
            .await?;
        manager
            .drop_table(
                Table::drop()
                    .table(AlgoFrameModelVersion::Table)
                    .if_exists()
                    .to_owned(),
            )
            .await?;
        manager
            .drop_table(Table::drop().table(AlgoFrameOutcome::Table).if_exists().to_owned())
            .await?;
        manager
            .drop_table(Table::drop().table(AlgoFrameDecision::Table).if_exists().to_owned())
            .await?;
        manager
            .drop_table(
                Table::drop()
                    .table(AlgoFrameMarketSnapshot::Table)
                    .if_exists()
                    .to_owned(),
            )
            .await?;
        Ok(())
    }
}

async fn create_market_snapshot(manager: &SchemaManager<'_>) -> Result<(), DbErr> {
    manager
        .create_table(
            Table::create()
                .table(AlgoFrameMarketSnapshot::Table)
                .if_not_exists()
                .col(
                    ColumnDef::new(AlgoFrameMarketSnapshot::Id)
                        .string()
                        .not_null()
                        .primary_key(),
                )
                .col(ColumnDef::new(AlgoFrameMarketSnapshot::ItemKey).string().not_null())
                .col(ColumnDef::new(AlgoFrameMarketSnapshot::WfmId).string().not_null())
                .col(ColumnDef::new(AlgoFrameMarketSnapshot::ItemName).string().not_null())
                .col(ColumnDef::new(AlgoFrameMarketSnapshot::Category).string().not_null())
                .col(ColumnDef::new(AlgoFrameMarketSnapshot::CreatedAt).string().not_null())
                .col(ColumnDef::new(AlgoFrameMarketSnapshot::BestBid).big_integer().not_null())
                .col(ColumnDef::new(AlgoFrameMarketSnapshot::BestAsk).big_integer().not_null())
                .col(ColumnDef::new(AlgoFrameMarketSnapshot::MidPrice).double().not_null())
                .col(ColumnDef::new(AlgoFrameMarketSnapshot::Spread).double().not_null())
                .col(ColumnDef::new(AlgoFrameMarketSnapshot::BuyDepth).big_integer().not_null())
                .col(ColumnDef::new(AlgoFrameMarketSnapshot::SellDepth).big_integer().not_null())
                .col(ColumnDef::new(AlgoFrameMarketSnapshot::Quality).double().not_null())
                .col(ColumnDef::new(AlgoFrameMarketSnapshot::AnomalyScore).double().not_null())
                .col(ColumnDef::new(AlgoFrameMarketSnapshot::Regime).string().not_null())
                .col(ColumnDef::new(AlgoFrameMarketSnapshot::Granularity).string().not_null())
                .col(ColumnDef::new(AlgoFrameMarketSnapshot::Payload).text().not_null())
                .to_owned(),
        )
        .await?;

    manager
        .create_index(
            Index::create()
                .name("idx_algoframe_snapshot_item_time")
                .table(AlgoFrameMarketSnapshot::Table)
                .col(AlgoFrameMarketSnapshot::ItemKey)
                .col(AlgoFrameMarketSnapshot::CreatedAt)
                .if_not_exists()
                .to_owned(),
        )
        .await?;

    manager
        .create_index(
            Index::create()
                .name("idx_algoframe_snapshot_granularity_time")
                .table(AlgoFrameMarketSnapshot::Table)
                .col(AlgoFrameMarketSnapshot::Granularity)
                .col(AlgoFrameMarketSnapshot::CreatedAt)
                .if_not_exists()
                .to_owned(),
        )
        .await?;

    Ok(())
}

async fn create_decision(manager: &SchemaManager<'_>) -> Result<(), DbErr> {
    manager
        .create_table(
            Table::create()
                .table(AlgoFrameDecision::Table)
                .if_not_exists()
                .col(
                    ColumnDef::new(AlgoFrameDecision::Id)
                        .string()
                        .not_null()
                        .primary_key(),
                )
                .col(ColumnDef::new(AlgoFrameDecision::SnapshotId).string().not_null())
                .col(ColumnDef::new(AlgoFrameDecision::ItemKey).string().not_null())
                .col(ColumnDef::new(AlgoFrameDecision::WfmId).string().not_null())
                .col(ColumnDef::new(AlgoFrameDecision::ItemName).string().not_null())
                .col(ColumnDef::new(AlgoFrameDecision::Category).string().not_null())
                .col(ColumnDef::new(AlgoFrameDecision::Side).string().not_null())
                .col(ColumnDef::new(AlgoFrameDecision::Status).string().not_null())
                .col(ColumnDef::new(AlgoFrameDecision::Lifecycle).string().not_null())
                .col(ColumnDef::new(AlgoFrameDecision::ChosenAction).string().not_null())
                .col(ColumnDef::new(AlgoFrameDecision::ChosenPropensity).double().not_null())
                .col(ColumnDef::new(AlgoFrameDecision::Price).big_integer().not_null())
                .col(ColumnDef::new(AlgoFrameDecision::Quantity).big_integer().not_null())
                .col(ColumnDef::new(AlgoFrameDecision::FilledQuantity).big_integer().not_null())
                .col(ColumnDef::new(AlgoFrameDecision::Capital).double().not_null())
                .col(ColumnDef::new(AlgoFrameDecision::PredictedProfit).double().not_null())
                .col(ColumnDef::new(AlgoFrameDecision::PredictedReward).double().not_null())
                .col(ColumnDef::new(AlgoFrameDecision::ActualProfit).double().not_null())
                .col(ColumnDef::new(AlgoFrameDecision::ActualReward).double().not_null())
                .col(ColumnDef::new(AlgoFrameDecision::ModelVersion).string().not_null())
                .col(ColumnDef::new(AlgoFrameDecision::ModelRole).string().not_null())
                .col(ColumnDef::new(AlgoFrameDecision::PolicyName).string().not_null())
                .col(
                    ColumnDef::new(AlgoFrameDecision::FeatureSchemaVersion)
                        .integer()
                        .not_null(),
                )
                .col(ColumnDef::new(AlgoFrameDecision::RewardVersion).integer().not_null())
                .col(ColumnDef::new(AlgoFrameDecision::PolicyVersion).integer().not_null())
                .col(ColumnDef::new(AlgoFrameDecision::SettingsHash).string().not_null())
                .col(ColumnDef::new(AlgoFrameDecision::CreatedAt).string().not_null())
                .col(ColumnDef::new(AlgoFrameDecision::UpdatedAt).string().not_null())
                .col(ColumnDef::new(AlgoFrameDecision::Payload).text().not_null())
                .to_owned(),
        )
        .await?;

    manager
        .create_index(
            Index::create()
                .name("idx_algoframe_decision_item_time")
                .table(AlgoFrameDecision::Table)
                .col(AlgoFrameDecision::ItemKey)
                .col(AlgoFrameDecision::UpdatedAt)
                .if_not_exists()
                .to_owned(),
        )
        .await?;

    manager
        .create_index(
            Index::create()
                .name("idx_algoframe_decision_status_time")
                .table(AlgoFrameDecision::Table)
                .col(AlgoFrameDecision::Status)
                .col(AlgoFrameDecision::UpdatedAt)
                .if_not_exists()
                .to_owned(),
        )
        .await?;

    Ok(())
}

async fn create_outcome(manager: &SchemaManager<'_>) -> Result<(), DbErr> {
    manager
        .create_table(
            Table::create()
                .table(AlgoFrameOutcome::Table)
                .if_not_exists()
                .col(
                    ColumnDef::new(AlgoFrameOutcome::Id)
                        .string()
                        .not_null()
                        .primary_key(),
                )
                .col(ColumnDef::new(AlgoFrameOutcome::DecisionId).string().not_null())
                .col(ColumnDef::new(AlgoFrameOutcome::ItemKey).string().not_null())
                .col(ColumnDef::new(AlgoFrameOutcome::Side).string().not_null())
                .col(ColumnDef::new(AlgoFrameOutcome::OutcomeType).string().not_null())
                .col(ColumnDef::new(AlgoFrameOutcome::Quantity).big_integer().not_null())
                .col(ColumnDef::new(AlgoFrameOutcome::Profit).double().not_null())
                .col(ColumnDef::new(AlgoFrameOutcome::Reward).double().not_null())
                .col(ColumnDef::new(AlgoFrameOutcome::FillHours).double().not_null())
                .col(ColumnDef::new(AlgoFrameOutcome::CycleHours).double().not_null())
                .col(ColumnDef::new(AlgoFrameOutcome::Simulated).boolean().not_null())
                .col(ColumnDef::new(AlgoFrameOutcome::CreatedAt).string().not_null())
                .col(ColumnDef::new(AlgoFrameOutcome::Payload).text().not_null())
                .to_owned(),
        )
        .await?;

    manager
        .create_index(
            Index::create()
                .name("idx_algoframe_outcome_decision")
                .table(AlgoFrameOutcome::Table)
                .col(AlgoFrameOutcome::DecisionId)
                .if_not_exists()
                .to_owned(),
        )
        .await?;

    Ok(())
}

async fn create_model_version(manager: &SchemaManager<'_>) -> Result<(), DbErr> {
    manager
        .create_table(
            Table::create()
                .table(AlgoFrameModelVersion::Table)
                .if_not_exists()
                .col(
                    ColumnDef::new(AlgoFrameModelVersion::Id)
                        .string()
                        .not_null()
                        .primary_key(),
                )
                .col(ColumnDef::new(AlgoFrameModelVersion::Version).string().not_null())
                .col(ColumnDef::new(AlgoFrameModelVersion::Role).string().not_null())
                .col(ColumnDef::new(AlgoFrameModelVersion::Active).boolean().not_null())
                .col(ColumnDef::new(AlgoFrameModelVersion::AverageReward).double().not_null())
                .col(ColumnDef::new(AlgoFrameModelVersion::FailureRate).double().not_null())
                .col(ColumnDef::new(AlgoFrameModelVersion::PredictionMae).double().not_null())
                .col(ColumnDef::new(AlgoFrameModelVersion::CalibrationError).double().not_null())
                .col(ColumnDef::new(AlgoFrameModelVersion::Drawdown).double().not_null())
                .col(ColumnDef::new(AlgoFrameModelVersion::CreatedAt).string().not_null())
                .col(ColumnDef::new(AlgoFrameModelVersion::PromotedAt).string())
                .col(ColumnDef::new(AlgoFrameModelVersion::Payload).text().not_null())
                .to_owned(),
        )
        .await
}

async fn create_model_evaluation(manager: &SchemaManager<'_>) -> Result<(), DbErr> {
    manager
        .create_table(
            Table::create()
                .table(AlgoFrameModelEvaluation::Table)
                .if_not_exists()
                .col(
                    ColumnDef::new(AlgoFrameModelEvaluation::Id)
                        .string()
                        .not_null()
                        .primary_key(),
                )
                .col(ColumnDef::new(AlgoFrameModelEvaluation::PolicyName).string().not_null())
                .col(ColumnDef::new(AlgoFrameModelEvaluation::SampleCount).big_integer().not_null())
                .col(ColumnDef::new(AlgoFrameModelEvaluation::AverageReward).double().not_null())
                .col(ColumnDef::new(AlgoFrameModelEvaluation::FailureRate).double().not_null())
                .col(ColumnDef::new(AlgoFrameModelEvaluation::IpsReward).double().not_null())
                .col(ColumnDef::new(AlgoFrameModelEvaluation::DrReward).double().not_null())
                .col(ColumnDef::new(AlgoFrameModelEvaluation::CreatedAt).string().not_null())
                .col(ColumnDef::new(AlgoFrameModelEvaluation::Payload).text().not_null())
                .to_owned(),
        )
        .await
}

async fn create_event(manager: &SchemaManager<'_>) -> Result<(), DbErr> {
    manager
        .create_table(
            Table::create()
                .table(AlgoFrameEvent::Table)
                .if_not_exists()
                .col(
                    ColumnDef::new(AlgoFrameEvent::Id)
                        .string()
                        .not_null()
                        .primary_key(),
                )
                .col(ColumnDef::new(AlgoFrameEvent::Kind).string().not_null())
                .col(ColumnDef::new(AlgoFrameEvent::Title).string().not_null())
                .col(ColumnDef::new(AlgoFrameEvent::Impact).double().not_null())
                .col(ColumnDef::new(AlgoFrameEvent::Confidence).double().not_null())
                .col(ColumnDef::new(AlgoFrameEvent::StartsAt).string().not_null())
                .col(ColumnDef::new(AlgoFrameEvent::EndsAt).string())
                .col(ColumnDef::new(AlgoFrameEvent::Source).string().not_null())
                .col(ColumnDef::new(AlgoFrameEvent::Payload).text().not_null())
                .to_owned(),
        )
        .await
}

async fn create_graph_edge(manager: &SchemaManager<'_>) -> Result<(), DbErr> {
    manager
        .create_table(
            Table::create()
                .table(AlgoFrameGraphEdge::Table)
                .if_not_exists()
                .col(
                    ColumnDef::new(AlgoFrameGraphEdge::Id)
                        .string()
                        .not_null()
                        .primary_key(),
                )
                .col(ColumnDef::new(AlgoFrameGraphEdge::FromKey).string().not_null())
                .col(ColumnDef::new(AlgoFrameGraphEdge::ToKey).string().not_null())
                .col(ColumnDef::new(AlgoFrameGraphEdge::Relation).string().not_null())
                .col(ColumnDef::new(AlgoFrameGraphEdge::Quantity).double().not_null())
                .col(ColumnDef::new(AlgoFrameGraphEdge::Cost).double().not_null())
                .col(ColumnDef::new(AlgoFrameGraphEdge::Payload).text().not_null())
                .to_owned(),
        )
        .await
}

async fn create_alert(manager: &SchemaManager<'_>) -> Result<(), DbErr> {
    manager
        .create_table(
            Table::create()
                .table(AlgoFrameAlert::Table)
                .if_not_exists()
                .col(
                    ColumnDef::new(AlgoFrameAlert::Id)
                        .string()
                        .not_null()
                        .primary_key(),
                )
                .col(ColumnDef::new(AlgoFrameAlert::Severity).string().not_null())
                .col(ColumnDef::new(AlgoFrameAlert::Code).string().not_null())
                .col(ColumnDef::new(AlgoFrameAlert::Message).text().not_null())
                .col(ColumnDef::new(AlgoFrameAlert::ItemKey).string())
                .col(ColumnDef::new(AlgoFrameAlert::CreatedAt).string().not_null())
                .col(ColumnDef::new(AlgoFrameAlert::Acknowledged).boolean().not_null())
                .col(ColumnDef::new(AlgoFrameAlert::Payload).text().not_null())
                .to_owned(),
        )
        .await
}

async fn create_setting(manager: &SchemaManager<'_>) -> Result<(), DbErr> {
    manager
        .create_table(
            Table::create()
                .table(AlgoFrameSetting::Table)
                .if_not_exists()
                .col(
                    ColumnDef::new(AlgoFrameSetting::Key)
                        .string()
                        .not_null()
                        .primary_key(),
                )
                .col(ColumnDef::new(AlgoFrameSetting::Value).text().not_null())
                .col(ColumnDef::new(AlgoFrameSetting::UpdatedAt).string().not_null())
                .to_owned(),
        )
        .await
}

#[derive(DeriveIden)]
enum AlgoFrameMarketSnapshot {
    #[sea_orm(iden = "algoframe_market_snapshot")]
    Table,
    Id,
    ItemKey,
    WfmId,
    ItemName,
    Category,
    CreatedAt,
    BestBid,
    BestAsk,
    MidPrice,
    Spread,
    BuyDepth,
    SellDepth,
    Quality,
    AnomalyScore,
    Regime,
    Granularity,
    Payload,
}

#[derive(DeriveIden)]
enum AlgoFrameDecision {
    #[sea_orm(iden = "algoframe_decision")]
    Table,
    Id,
    SnapshotId,
    ItemKey,
    WfmId,
    ItemName,
    Category,
    Side,
    Status,
    Lifecycle,
    ChosenAction,
    ChosenPropensity,
    Price,
    Quantity,
    FilledQuantity,
    Capital,
    PredictedProfit,
    PredictedReward,
    ActualProfit,
    ActualReward,
    ModelVersion,
    ModelRole,
    PolicyName,
    FeatureSchemaVersion,
    RewardVersion,
    PolicyVersion,
    SettingsHash,
    CreatedAt,
    UpdatedAt,
    Payload,
}

#[derive(DeriveIden)]
enum AlgoFrameOutcome {
    #[sea_orm(iden = "algoframe_outcome")]
    Table,
    Id,
    DecisionId,
    ItemKey,
    Side,
    OutcomeType,
    Quantity,
    Profit,
    Reward,
    FillHours,
    CycleHours,
    Simulated,
    CreatedAt,
    Payload,
}

#[derive(DeriveIden)]
enum AlgoFrameModelVersion {
    #[sea_orm(iden = "algoframe_model_version")]
    Table,
    Id,
    Version,
    Role,
    Active,
    AverageReward,
    FailureRate,
    PredictionMae,
    CalibrationError,
    Drawdown,
    CreatedAt,
    PromotedAt,
    Payload,
}

#[derive(DeriveIden)]
enum AlgoFrameModelEvaluation {
    #[sea_orm(iden = "algoframe_model_evaluation")]
    Table,
    Id,
    PolicyName,
    SampleCount,
    AverageReward,
    FailureRate,
    IpsReward,
    DrReward,
    CreatedAt,
    Payload,
}

#[derive(DeriveIden)]
enum AlgoFrameEvent {
    #[sea_orm(iden = "algoframe_event")]
    Table,
    Id,
    Kind,
    Title,
    Impact,
    Confidence,
    StartsAt,
    EndsAt,
    Source,
    Payload,
}

#[derive(DeriveIden)]
enum AlgoFrameGraphEdge {
    #[sea_orm(iden = "algoframe_graph_edge")]
    Table,
    Id,
    FromKey,
    ToKey,
    Relation,
    Quantity,
    Cost,
    Payload,
}

#[derive(DeriveIden)]
enum AlgoFrameAlert {
    #[sea_orm(iden = "algoframe_alert")]
    Table,
    Id,
    Severity,
    Code,
    Message,
    ItemKey,
    CreatedAt,
    Acknowledged,
    Payload,
}

#[derive(DeriveIden)]
enum AlgoFrameSetting {
    #[sea_orm(iden = "algoframe_setting")]
    Table,
    Key,
    Value,
    UpdatedAt,
}
