"""initial schema

Revision ID: f82f1648cbbe
Revises: 
Create Date: 2026-07-27 18:14:17.759603

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa



revision: str = 'f82f1648cbbe'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:

    op.create_table('users',
    sa.Column('id', sa.String(length=36), nullable=False),
    sa.Column('email', sa.String(length=255), nullable=False),
    sa.Column('hashed_password', sa.String(length=255), nullable=False),
    sa.Column('full_name', sa.String(length=255), nullable=True),
    sa.Column('role', sa.String(length=32), nullable=False),
    sa.Column('is_active', sa.Boolean(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('last_login_at', sa.DateTime(timezone=True), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_users_email'), 'users', ['email'], unique=True)
    op.create_table('audit_logs',
    sa.Column('id', sa.String(length=36), nullable=False),
    sa.Column('user_id', sa.String(length=36), nullable=True),
    sa.Column('action', sa.String(length=100), nullable=False),
    sa.Column('resource_id', sa.String(length=36), nullable=True),
    sa.Column('resource_type', sa.String(length=50), nullable=True),
    sa.Column('ip_address', sa.String(length=45), nullable=True),
    sa.Column('user_agent', sa.String(length=500), nullable=True),
    sa.Column('success', sa.Boolean(), nullable=False),
    sa.Column('error_detail', sa.Text(), nullable=True),
    sa.Column('metadata', sa.JSON(), nullable=True),
    sa.Column('timestamp', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='SET NULL'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_audit_logs_action'), 'audit_logs', ['action'], unique=False)
    op.create_index(op.f('ix_audit_logs_resource_id'), 'audit_logs', ['resource_id'], unique=False)
    op.create_index(op.f('ix_audit_logs_timestamp'), 'audit_logs', ['timestamp'], unique=False)
    op.create_index(op.f('ix_audit_logs_user_id'), 'audit_logs', ['user_id'], unique=False)
    op.create_table('model_versions',
    sa.Column('id', sa.String(length=36), nullable=False),
    sa.Column('group_key', sa.String(length=50), nullable=False),
    sa.Column('version_tag', sa.String(length=50), nullable=False),
    sa.Column('architecture', sa.String(length=100), nullable=False),
    sa.Column('weights_path', sa.String(length=500), nullable=False),
    sa.Column('val_accuracy', sa.Float(), nullable=True),
    sa.Column('val_auc', sa.Float(), nullable=True),
    sa.Column('val_sensitivity', sa.JSON(), nullable=True),
    sa.Column('val_specificity', sa.JSON(), nullable=True),
    sa.Column('val_set_description', sa.Text(), nullable=True),
    sa.Column('calibration_temperature', sa.Float(), nullable=True),
    sa.Column('calibration_ece', sa.Float(), nullable=True),
    sa.Column('active', sa.Boolean(), nullable=False),
    sa.Column('registered_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('registered_by', sa.String(length=36), nullable=True),
    sa.ForeignKeyConstraint(['registered_by'], ['users.id'], ondelete='SET NULL'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_model_versions_active'), 'model_versions', ['active'], unique=False)
    op.create_index(op.f('ix_model_versions_group_key'), 'model_versions', ['group_key'], unique=False)
    op.create_table('scan_results',
    sa.Column('id', sa.String(length=36), nullable=False),
    sa.Column('user_id', sa.String(length=36), nullable=True),
    sa.Column('diagnosis', sa.String(length=100), nullable=False),
    sa.Column('confidence', sa.Float(), nullable=False),
    sa.Column('group_name', sa.String(length=100), nullable=True),
    sa.Column('probabilities', sa.JSON(), nullable=True),
    sa.Column('calibrated', sa.Boolean(), nullable=False),
    sa.Column('calibration_temperature', sa.Float(), nullable=True),
    sa.Column('uncertainty', sa.Float(), nullable=True),
    sa.Column('requires_human_review', sa.Boolean(), nullable=False),
    sa.Column('review_reasons', sa.JSON(), nullable=True),
    sa.Column('icd10_code', sa.String(length=20), nullable=True),
    sa.Column('snomed_code', sa.String(length=20), nullable=True),
    sa.Column('urgency', sa.String(length=32), nullable=True),
    sa.Column('urgency_rank', sa.Integer(), nullable=True),
    sa.Column('hybrid_warnings', sa.JSON(), nullable=True),
    sa.Column('hybrid_warnings_structured', sa.JSON(), nullable=True),
    sa.Column('iqa_acceptable', sa.Boolean(), nullable=True),
    sa.Column('iqa_warnings', sa.JSON(), nullable=True),
    sa.Column('symptoms_reported', sa.JSON(), nullable=True),
    sa.Column('model_version_id', sa.String(length=36), nullable=True),
    sa.Column('router_group_idx', sa.Integer(), nullable=True),
    sa.Column('image_path', sa.String(length=500), nullable=True),
    sa.Column('heatmap_path', sa.String(length=500), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['model_version_id'], ['model_versions.id'], ondelete='SET NULL'),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='SET NULL'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_scan_results_diagnosis'), 'scan_results', ['diagnosis'], unique=False)
    op.create_index(op.f('ix_scan_results_user_id'), 'scan_results', ['user_id'], unique=False)
    op.create_table('clinician_overrides',
    sa.Column('id', sa.String(length=36), nullable=False),
    sa.Column('scan_id', sa.String(length=36), nullable=False),
    sa.Column('clinician_id', sa.String(length=36), nullable=True),
    sa.Column('verdict', sa.String(length=32), nullable=False),
    sa.Column('corrected_diagnosis', sa.String(length=100), nullable=True),
    sa.Column('corrected_icd10', sa.String(length=20), nullable=True),
    sa.Column('notes', sa.Text(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['clinician_id'], ['users.id'], ondelete='SET NULL'),
    sa.ForeignKeyConstraint(['scan_id'], ['scan_results.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('scan_id')
    )



def downgrade() -> None:

    op.drop_table('clinician_overrides')
    op.drop_index(op.f('ix_scan_results_user_id'), table_name='scan_results')
    op.drop_index(op.f('ix_scan_results_diagnosis'), table_name='scan_results')
    op.drop_table('scan_results')
    op.drop_index(op.f('ix_model_versions_group_key'), table_name='model_versions')
    op.drop_index(op.f('ix_model_versions_active'), table_name='model_versions')
    op.drop_table('model_versions')
    op.drop_index(op.f('ix_audit_logs_user_id'), table_name='audit_logs')
    op.drop_index(op.f('ix_audit_logs_timestamp'), table_name='audit_logs')
    op.drop_index(op.f('ix_audit_logs_resource_id'), table_name='audit_logs')
    op.drop_index(op.f('ix_audit_logs_action'), table_name='audit_logs')
    op.drop_table('audit_logs')
    op.drop_index(op.f('ix_users_email'), table_name='users')
    op.drop_table('users')

